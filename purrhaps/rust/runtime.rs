#![allow(dead_code)] // Each generated program uses a different runtime subset.
use serde_json::{json, Value as J};
#[path = "project.rs"]
mod project;
#[path = "providers.rs"]
mod providers;
use project::Project;
use std::{
    cell::RefCell,
    collections::HashMap,
    io::{Read, Write},
    process::{Command, Stdio},
    rc::Rc,
    time::{Duration, Instant},
};

type Scope = usize;
pub type Res<T> = Result<T, Fault>;
pub type Expr = fn(&mut Rt, Scope) -> Res<V>;
pub type Block = fn(&mut Rt, Scope) -> Res<Option<V>>;
type Recipe = Rc<dyn Fn(&mut Rt, Scope) -> Res<Sample>>;
#[derive(Clone, Debug)]
pub struct Fault {
    pub kind: String,
    pub message: String,
    pub line: Option<usize>,
}
fn fail<T>(kind: &str, message: impl Into<String>) -> Res<T> {
    Err(Fault {
        kind: kind.into(),
        message: message.into(),
        line: None,
    })
}
pub fn located<T>(result: Res<T>, line: usize) -> Res<T> {
    result.map_err(|mut e| {
        if e.line.is_none() {
            e.line = Some(line);
        }
        e
    })
}
#[derive(Clone)]
pub struct Sample {
    data: J,
    confidence: f64,
    // Terminal random selection grants a gate permission, never confidence.
    gate_confidence: f64,
    resolution: Option<J>,
    // History includes rejected attempts, so it cannot inherit that permission.
    resolution_gate_confidence: Option<f64>,
    because: String,
    citations: Vec<usize>,
    generative: bool,
    decision: Option<J>,
}
impl Sample {
    fn new(data: J) -> Self {
        Self {
            data,
            confidence: 1.,
            gate_confidence: 1.,
            resolution: None,
            resolution_gate_confidence: None,
            because: "A literal or deterministic computation.".into(),
            citations: vec![],
            generative: false,
            decision: None,
        }
    }
    fn explained(data: J, because: &str) -> Self {
        let mut v = Self::new(data);
        v.because = because.into();
        v
    }
}
#[derive(Clone)]
pub enum V {
    Certain(Sample),
    Maybe(Rc<RefCell<Uncertain>>),
}
pub struct Uncertain {
    recipe: Recipe,
    cached: bool,
    cache: HashMap<Scope, Sample>,
    owner: Option<String>,
}
fn maybe(recipe: impl Fn(&mut Rt, Scope) -> Res<Sample> + 'static, cached: bool) -> V {
    V::Maybe(Rc::new(RefCell::new(Uncertain {
        recipe: Rc::new(recipe),
        cached,
        cache: HashMap::new(),
        owner: None,
    })))
}
#[derive(Clone)]
struct Function {
    params: Vec<(String, String)>,
    typ: String,
    body: Block,
    closure: Scope,
}
#[derive(Clone)]
enum Binding {
    Value(V),
    Function(Function),
}
struct Env {
    label: String,
    parent: Option<Scope>,
    values: HashMap<String, Binding>,
}
pub struct Rt {
    envs: Vec<Env>,
    source: String,
    pub calls: usize,
    temperature: f64,
    temperature_depth: usize,
    fallback_profile_temperature: Option<f64>,
    seed: u64,
    route_rng: HashMap<String, u64>,
    project: Option<Project>,
    generative_route: Option<String>,
    decision_route: Option<String>,
    timeout_override: Option<Duration>,
    max_calls: usize,
    rng: u64,
    escalation_rng: u64,
    command: Vec<String>,
    decision_command: Vec<String>,
    timeout: Duration,
    depth: usize,
}
fn citations(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = a.to_vec();
    for n in b {
        if !out.contains(n) {
            out.push(*n)
        }
    }
    out
}
fn typename(v: &J) -> &str {
    match v {
        J::Null => "NoneType",
        J::Bool(_) => "bool",
        J::Number(n) if n.is_i64() || n.is_u64() => "int",
        J::Number(_) => "float",
        J::String(_) => "str",
        J::Array(_) => "list",
        J::Object(_) => "dict",
    }
}
fn type_parts(typ: &str) -> (&str, Vec<&str>) {
    if let Some(start) = typ.find('<') {
        if !typ.ends_with('>') {
            return (typ, vec![]);
        }
        let inner = &typ[start + 1..typ.len() - 1];
        let mut depth = 0;
        let mut last = 0;
        let mut args = vec![];
        for (i, c) in inner.char_indices() {
            match c {
                '<' => depth += 1,
                '>' => depth -= 1,
                ',' if depth == 0 => {
                    args.push(inner[last..i].trim());
                    last = i + 1;
                }
                _ => (),
            }
        }
        args.push(inner[last..].trim());
        (&typ[..start], args)
    } else {
        (typ, vec![])
    }
}
fn shape(data: &J, typ: &str) -> Res<bool> {
    if let Some(inner) = typ.strip_suffix('?') {
        // Validate the annotation even for null.
        let valid = shape(data, inner)?;
        return Ok(data.is_null() || valid);
    }
    let (name, args) = type_parts(typ);
    let record = |fields: &[(&str, &str)]| -> Res<bool> {
        let Some(map) = data.as_object() else {
            return Ok(false);
        };
        if map.len() != fields.len() {
            return Ok(false);
        }
        for (key, typ) in fields {
            let Some(value) = map.get(*key) else {
                return Ok(false);
            };
            if !shape(value, typ)? {
                return Ok(false);
            }
        }
        Ok(true)
    };
    if !args.is_empty() && !["List", "Map", "Pair"].contains(&name) {
        return fail("TypeError", format!("unknown type '{typ}'"));
    }
    Ok(match name {
        "" | "Any" | "Json" => finite_json(data),
        "Int" | "Long" => data.is_i64(),
        "Float" | "Double" => data.is_f64(),
        "String" => data.is_string(),
        "Bool" | "Boolean" => data.is_boolean(),
        "Unit" => data.is_null(),
        "Human" => record(&[("lastMessage", "String")])?,
        "List" => {
            if args.len() > 1 {
                return fail("TypeError", "List requires one type argument");
            }
            if let Some(items) = data.as_array() {
                let mut valid = true;
                for item in items {
                    valid &= shape(item, args.first().copied().unwrap_or("Json"))?;
                }
                valid
            } else {
                false
            }
        }
        "Map" => {
            if !args.is_empty() && (args.len() != 2 || args[0] != "String") {
                return fail("TypeError", "Map requires String keys and a value type");
            }
            if let Some(items) = data.as_object() {
                let mut valid = true;
                for item in items.values() {
                    valid &= shape(item, args.get(1).copied().unwrap_or("Json"))?;
                }
                valid
            } else {
                false
            }
        }
        "Pair" => {
            if !args.is_empty() && args.len() != 2 {
                return fail("TypeError", "Pair requires two type arguments");
            }
            record(&[
                ("first", args.first().copied().unwrap_or("Json")),
                ("second", args.get(1).copied().unwrap_or("Json")),
            ])?
        }
        "Message" => {
            record(&[("role", "String"), ("content", "String")])?
                && ["system", "user", "assistant", "tool"]
                    .contains(&data["role"].as_str().unwrap_or(""))
        }
        "Prompt" => {
            record(&[("messages", "List<Message>")])?
                && !data["messages"].as_array().unwrap().is_empty()
        }
        "DecisionInput" => {
            let valid = record(&[
                ("question", "String"),
                ("options", "List<Json>"),
                ("context", "Json"),
            ])?;
            if !valid {
                false
            } else {
                let options = data["options"].as_array().unwrap();
                !data["question"].as_str().unwrap().trim().is_empty()
                    && options.len() >= 2
                    && options
                        .iter()
                        .enumerate()
                        .all(|(i, a)| !options[..i].iter().any(|b| strict_equal(a, b)))
            }
        }
        "TokenUsage" => {
            record(&[("inputTokens", "Int"), ("outputTokens", "Int")])?
                && data["inputTokens"].as_i64().unwrap() >= 0
                && data["outputTokens"].as_i64().unwrap() >= 0
        }
        "ModelResponse" => record(&[
            ("text", "String"),
            ("finishReason", "String"),
            ("usage", "TokenUsage"),
        ])?,
        _ => return fail("TypeError", format!("unknown type '{typ}'")),
    })
}
fn finite_json(v: &J) -> bool {
    match v {
        J::Number(n) => n.as_f64().is_some_and(f64::is_finite),
        J::Array(a) => a.iter().all(finite_json),
        J::Object(o) => o.values().all(finite_json),
        _ => true,
    }
}
fn strict_equal(a: &J, b: &J) -> bool {
    match (a, b) {
        (J::Number(a), J::Number(b)) => {
            if a.is_f64() != b.is_f64() {
                false
            } else {
                a == b
            }
        }
        (J::Array(a), J::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| strict_equal(a, b))
        }
        (J::Object(a), J::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, a)| b.get(k).is_some_and(|b| strict_equal(a, b)))
        }
        _ => a == b,
    }
}
fn validate_type(typ: &str) -> Res<()> {
    if let Some(inner) = typ.strip_suffix('?') {
        return validate_type(inner);
    }
    let (name, args) = type_parts(typ);
    let arity = match name {
        "List" => Some(1),
        "Map" | "Pair" => Some(2),
        "" | "Any" | "Json" | "Int" | "Long" | "Float" | "Double" | "String" | "Bool"
        | "Boolean" | "Unit" | "Human" | "Message" | "Prompt" | "DecisionInput" | "TokenUsage"
        | "ModelResponse" => None,
        _ => return fail("TypeError", format!("unknown type '{typ}'")),
    };
    if !args.is_empty() && arity != Some(args.len()) {
        return fail("TypeError", format!("invalid type arguments for {name}"));
    }
    if name == "Map" && !args.is_empty() && args[0] != "String" {
        return fail("TypeError", "Map keys must be String");
    }
    for arg in args {
        validate_type(arg)?;
    }
    Ok(())
}
fn typed(v: Sample, typ: &str) -> Res<Sample> {
    validate_type(typ)?;
    if shape(&v.data, typ)? {
        Ok(v)
    } else {
        fail(
            "TypeError",
            format!("expected {typ}, got {}", typename(&v.data)),
        )
    }
}
pub fn literal(data: J, line: usize) -> V {
    V::Certain(Sample {
        data,
        confidence: 1.,
        gate_confidence: 1.,
        resolution: None,
        resolution_gate_confidence: None,
        because: format!("Literal on line {line}."),
        citations: vec![line],
        generative: false,
        decision: None,
    })
}
fn definite(data: J) -> V {
    V::Certain(Sample::new(data))
}
fn float_display(x: f64) -> String {
    let magnitude = x.abs();
    if magnitude != 0. && !(1e-4..1e16).contains(&magnitude) {
        let scientific = format!("{x:e}");
        let (mantissa, exponent) = scientific.split_once('e').unwrap();
        let exponent: i32 = exponent.parse().unwrap();
        format!("{mantissa}e{exponent:+03}")
    } else if x.fract() == 0. {
        format!("{x:.1}")
    } else {
        x.to_string()
    }
}
fn json_display(v: &J) -> String {
    match v {
        J::Number(n) if n.is_f64() => float_display(n.as_f64().unwrap()),
        J::Array(items) => format!(
            "[{}]",
            items.iter().map(json_display).collect::<Vec<_>>().join(",")
        ),
        J::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(key, value)| format!("{}:{}", json!(key), json_display(value)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        _ => v.to_string(),
    }
}
fn display(v: &J) -> String {
    match v {
        J::String(text) => text.clone(),
        _ => json_display(v),
    }
}
fn trim(text: &str) -> &str {
    // Python additionally treats these four ASCII separators as whitespace.
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}
impl Rt {
    pub fn new(
        source: &str,
        seed: u64,
        max_calls: usize,
        command: Vec<String>,
        timeout: Duration,
    ) -> Self {
        let mut values = HashMap::new();
        values.insert(
            "this".into(),
            Binding::Value(definite(json!({"self":"purrhaps"}))),
        );
        Self {
            envs: vec![Env {
                label: "module".into(),
                parent: None,
                values,
            }],
            source: source.into(),
            calls: 0,
            temperature: 0.7,
            temperature_depth: 0,
            fallback_profile_temperature: None,
            seed,
            route_rng: HashMap::new(),
            project: None,
            generative_route: None,
            decision_route: None,
            timeout_override: None,
            max_calls,
            rng: seed,
            escalation_rng: seed,
            command,
            decision_command: vec![],
            timeout,
            depth: 0,
        }
    }
    fn enter(&mut self) -> Res<()> {
        if self.depth >= 128 {
            return fail(
                "PurrhapsError",
                "the cat chased its own tail: recursion limit exceeded",
            );
        }
        self.depth += 1;
        Ok(())
    }
    pub fn eval(&mut self, expression: Expr, scope: Scope) -> Res<V> {
        self.enter()?;
        let result = expression(self, scope);
        self.depth -= 1;
        result
    }
    pub fn child(&mut self, parent: Scope, label: &str) -> Scope {
        let n = self.envs.len();
        self.envs.push(Env {
            label: format!("{label}#{n}"),
            parent: Some(parent),
            values: HashMap::new(),
        });
        n
    }
    fn lookup(&self, mut scope: Scope, name: &str) -> Res<Binding> {
        loop {
            if let Some(v) = self.envs[scope].values.get(name) {
                return Ok(v.clone());
            }
            match self.envs[scope].parent {
                Some(p) => scope = p,
                None => return fail("NameError", format!("unknown name '{name}'")),
            }
        }
    }
    fn define(&mut self, scope: Scope, name: &str, value: Binding) -> Res<()> {
        if self.envs[scope].values.contains_key(name) {
            return fail(
                "NameError",
                format!("'{name}' is already declared in this scope"),
            );
        }
        self.envs[scope].values.insert(name.into(), value);
        Ok(())
    }
    pub fn name(&self, scope: Scope, name: &str) -> Res<V> {
        match self.lookup(scope, name)? {
            Binding::Value(v) => Ok(v),
            _ => fail("TypeError", "functions must be called"),
        }
    }
    pub fn force(&mut self, v: &V, scope: Scope) -> Res<Sample> {
        match v {
            V::Certain(s) => Ok(s.clone()),
            V::Maybe(cell) => {
                let (recipe, cached) = {
                    let mut m = cell.borrow_mut();
                    m.owner = Some(self.envs[scope].label.clone());
                    if let Some(v) = m.cache.get(&scope) {
                        return Ok(v.clone());
                    }
                    (m.recipe.clone(), m.cached)
                };
                self.enter()?;
                let result = recipe(self, scope);
                self.depth -= 1;
                let result = result?;
                if cached {
                    cell.borrow_mut().cache.insert(scope, result.clone());
                }
                Ok(result)
            }
        }
    }
    pub fn collapse(&mut self, v: &V, scope: Scope) -> Res<V> {
        let s = self.force(v, scope)?;
        if s.gate_confidence < 0.8 {
            return fail(
                "UncertaintyException",
                format!(
                    "confidence {:.2} is below 0.80{}",
                    s.confidence,
                    if s.generative && s.decision.is_none() {
                        format!(": {}", s.because)
                    } else {
                        String::new()
                    }
                ),
            );
        }
        Ok(V::Certain(s))
    }
    fn checked(&mut self, v: V, typ: &str) -> Res<V> {
        match v {
            V::Certain(s) => Ok(V::Certain(typed(s, typ)?)),
            _ => {
                let t = typ.to_owned();
                Ok(maybe(move |rt, s| typed(rt.force(&v, s)?, &t), false))
            }
        }
    }
    pub fn declare(
        &mut self,
        scope: Scope,
        mode: &str,
        name: &str,
        typ: &str,
        expr: Expr,
        line: usize,
    ) -> Res<()> {
        let v = if mode == "val" {
            let v = self.eval(expr, scope)?;
            self.checked(v, typ)?
        } else {
            let t = typ.to_owned();
            let decision_mode = mode == "definitely";
            maybe(
                move |rt, reader| {
                    let input = rt.eval(expr, scope)?;
                    let seed = rt.force(&input, reader)?;
                    let mut answer = if decision_mode {
                        if seed.data.as_object().is_some_and(|m| {
                            m.len() == 3
                                && ["question", "options", "context"]
                                    .iter()
                                    .all(|k| m.contains_key(*k))
                        }) {
                            typed(seed.clone(), "DecisionInput")?;
                            rt.ask_decision(
                                "decision",
                                seed.data.clone(),
                                &t,
                                line,
                                seed.data["options"].clone(),
                            )?
                        } else if seed.decision.is_some() {
                            return typed(seed, &t);
                        } else {
                            return fail("TypeError", "definitely requires a DecisionInput or an existing decision expression; meow with bounded options");
                        }
                    } else {
                        rt.ask("read", seed.data.clone(), &t, line)?
                    };
                    answer.generative |= seed.generative;
                    if answer.decision.is_none() {
                        answer.decision = seed.decision.clone();
                    }
                    answer.confidence = answer.confidence.min(seed.confidence);
                    answer.gate_confidence = answer.gate_confidence.min(seed.gate_confidence);
                    answer.resolution = seed.resolution.clone();
                    answer.resolution_gate_confidence = seed.resolution_gate_confidence;
                    answer.citations = if decision_mode {
                        citations(&answer.citations, &seed.citations)
                    } else {
                        citations(&seed.citations, &answer.citations)
                    };
                    Ok(answer)
                },
                mode == "perhaps",
            )
        };
        self.define(scope, name, Binding::Value(v))
    }
    pub fn escalate(
        &mut self,
        scope: Scope,
        name: &str,
        typ: &str,
        expr: Expr,
        line: usize,
    ) -> Res<()> {
        let typ = typ.to_owned();
        let value = maybe(
            move |rt, reader| {
                let input = rt.eval(expr, scope)?;
                let seed = typed(rt.force(&input, reader)?, "DecisionInput")?;
                validate_type(&typ)?;
                for option in seed.data["options"].as_array().unwrap() {
                    if !shape(option, &typ)? {
                        return fail(
                            "TypeError",
                            "every escalation option must match the requested type",
                        );
                    }
                }
                let config = rt
                    .project
                    .as_ref()
                    .map(|p| p.data["escalation"].clone())
                    .unwrap_or(J::Null);
                let previous_decision = rt.decision_route.clone();
                let previous_generative = rt.generative_route.clone();
                if let Some(route) = config["decision"].as_str() {
                    rt.decision_route = Some(route.into());
                }
                if let Some(route) = config["generative"].as_str() {
                    rt.generative_route = Some(route.into());
                }
                let result = rt.resolve_pipeline(seed, &typ, line, &config);
                rt.decision_route = previous_decision;
                rt.generative_route = previous_generative;
                result
            },
            false,
        );
        self.define(scope, name, Binding::Value(value))
    }
    fn resolve_pipeline(
        &mut self,
        seed: Sample,
        typ: &str,
        line: usize,
        config: &J,
    ) -> Res<Sample> {
        let threshold = config["threshold"].as_f64().unwrap_or(0.8);
        let exposure = config["exposure"].as_str().unwrap_or("blind");
        let options = seed.data["options"].clone();
        let mut attempts = vec![];
        let mut first =
            self.ask_decision("decision", seed.data.clone(), typ, line, options.clone())?;
        let effective = first.confidence.min(seed.confidence);
        let decision_evidence = first.decision.clone().unwrap();
        let mut history_gate = seed.gate_confidence.min(first.confidence);
        let scores = decision_evidence["scores"].clone();
        attempts.push(json!({"kind":"decision","route":self.decision_route,"value":first.data,"reportedConfidence":first.confidence,"effectiveConfidence":effective,"scores":scores}));
        if effective >= threshold {
            first.confidence = effective;
            first.gate_confidence = first.gate_confidence.min(seed.gate_confidence);
            first.generative |= seed.generative;
            first.citations = citations(&first.citations, &seed.citations);
            first.resolution_gate_confidence = Some(history_gate);
            first.resolution = Some(
                json!({"strategy":"decision","exposure":exposure,"forced":false,"threshold":threshold,"selectionProbability":null,"attempts":attempts,"decision":decision_evidence}),
            );
            return Ok(first);
        }
        let prior = if exposure == "informed" {
            Some(json!({"value":first.data,"confidence":first.confidence,"scores":scores}))
        } else {
            None
        };
        let mut final_answer =
            self.ask_resolve(seed.data.clone(), typ, line, options.clone(), prior)?;
        history_gate = history_gate.min(final_answer.confidence);
        let effective = final_answer.confidence.min(seed.confidence);
        attempts.push(json!({"kind":"generative","route":self.generative_route,"value":final_answer.data,"reportedConfidence":final_answer.confidence,"effectiveConfidence":effective,"scores":null}));
        let strategy;
        let probability;
        if effective >= threshold {
            strategy = "generative";
            probability = J::Null;
            final_answer.confidence = effective;
            final_answer.gate_confidence = final_answer.gate_confidence.min(seed.gate_confidence);
            final_answer.decision = seed.decision.clone();
        } else if config["on_low_confidence"].as_str().unwrap_or("error") == "random" {
            strategy = "random";
            let choices = options.as_array().unwrap();
            let selected = choices[self.random_index(choices.len())].clone();
            probability = json!(1. / choices.len() as f64);
            final_answer = Sample::new(selected.clone());
            final_answer.confidence = 0.;
            final_answer.gate_confidence = 1.;
            final_answer.because.clear();
            final_answer.decision = Some(
                json!({"kind":"random","input":seed.data,"options":options,"selected":selected,"scores":null}),
            );
        } else {
            return fail(
                "UncertaintyException",
                "escalation answers remain below the confidence threshold",
            );
        }
        final_answer.generative |= seed.generative;
        final_answer.citations = citations(&seed.citations, &final_answer.citations);
        final_answer.resolution_gate_confidence = Some(history_gate);
        final_answer.resolution = Some(
            json!({"strategy":strategy,"exposure":exposure,"forced":strategy == "random","threshold":threshold,"selectionProbability":probability,"attempts":attempts,"decision":decision_evidence}),
        );
        Ok(final_answer)
    }
    fn random_index(&mut self, count: usize) -> usize {
        let count = count as u64;
        let threshold = count.wrapping_neg() % count;
        loop {
            self.escalation_rng = self.escalation_rng.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = self.escalation_rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            if z >= threshold {
                return (z % count) as usize;
            }
        }
    }
    pub fn function(
        &mut self,
        scope: Scope,
        name: &str,
        params: Vec<(&str, &str)>,
        typ: &str,
        body: Block,
    ) -> Res<()> {
        let mut seen = std::collections::HashSet::new();
        for (p, _) in &params {
            if !seen.insert(*p) {
                return fail("NameError", "duplicate function parameter");
            }
        }
        self.define(
            scope,
            name,
            Binding::Function(Function {
                params: params
                    .into_iter()
                    .map(|(n, t)| (n.into(), t.into()))
                    .collect(),
                typ: typ.into(),
                body,
                closure: scope,
            }),
        )
    }
    pub fn branch(&mut self, scope: Scope, v: V, yes: Block, no: Block) -> Res<Option<V>> {
        let v = self.collapse(&v, scope)?;
        let sample = self.force(&v, scope)?;
        let b = typed(sample, "Bool")?.data.as_bool().unwrap();
        let child = self.child(scope, "branch");
        if b {
            yes(self, child)
        } else {
            no(self, child)
        }
    }
    pub fn temperature(&mut self, scope: Scope, v: V, body: Block) -> Res<Option<V>> {
        let sample = match v {
            V::Certain(v) => v,
            _ => return fail("TypeError", "temperature needs a definite number; use !!"),
        };
        let n = sample
            .data
            .as_f64()
            .filter(|n| n.is_finite() && (0.0..=2.0).contains(n))
            .ok_or_else(|| Fault {
                kind: "TypeError".into(),
                message: "temperature must be a number between 0 and 2".into(),
                line: None,
            })?;
        let previous = self.temperature;
        self.temperature = n;
        self.temperature_depth += 1;
        let child = self.child(scope, "temperature");
        let result = body(self, child);
        self.temperature = previous;
        self.temperature_depth -= 1;
        result
    }
    pub fn require_project(&self) -> Res<()> {
        if self.project.is_none() {
            return fail(
                "ConfigurationError",
                "with model requires a configured project",
            );
        }
        Ok(())
    }
    pub fn model(&mut self, scope: Scope, v: V, body: Block) -> Res<Option<V>> {
        self.require_project()?;
        let name = match v {
            V::Certain(s) => s.data.as_str().map(str::to_owned),
            _ => None,
        }
        .ok_or_else(|| Fault {
            kind: "TypeError".into(),
            message: "model route needs a definite String; use !!".into(),
            line: None,
        })?;
        let project = self.project.as_ref().ok_or_else(|| Fault {
            kind: "ConfigurationError".into(),
            message: "with model requires a configured project".into(),
            line: None,
        })?;
        let (_, model) = project.route(&name, None)?;
        let decision = model["kind"] == "decision";
        let previous = if decision {
            self.decision_route.replace(name)
        } else {
            self.generative_route.replace(name)
        };
        let child = self.child(scope, "model");
        let result = body(self, child);
        if decision {
            self.decision_route = previous;
        } else {
            self.generative_route = previous;
        }
        result
    }
    fn configured(&mut self, route: &str, mut request: J) -> Res<J> {
        let project = self.project.as_ref().unwrap();
        let (provider, model) = project.route(route, None)?;
        let timeout = project.timeout(&model, self.timeout_override);
        let root = project.root.clone();
        if self.temperature_depth == 0 {
            if let Some(temp) = model.get("temperature") {
                request["temperature"] = temp.clone();
            }
        }
        if provider["transport"] == "demo" {
            let previous = self.temperature;
            let previous_rng = self.rng;
            self.rng = self.route_rng.get(route).copied().unwrap_or(self.seed);
            self.temperature = request["temperature"].as_f64().unwrap();
            let kind = request["kind"].as_str().unwrap();
            let line = request["line"].as_u64().unwrap() as usize;
            let answer = if model["kind"] == "decision" {
                let value = if kind == "difference" {
                    let eq = self.demo("equivalence", request["input"].clone(), "Bool", line);
                    json!(!eq["value"].as_bool().unwrap())
                } else {
                    request["options"][0].clone()
                };
                json!({"value":value,"confidence":if kind=="difference" {0.94} else {0.92}})
            } else {
                self.demo(
                    kind,
                    request["input"].clone(),
                    request["type"].as_str().unwrap_or(""),
                    line,
                )
            };
            self.temperature = previous;
            self.route_rng.insert(route.into(), self.rng);
            self.rng = previous_rng;
            return Ok(answer);
        }
        providers::ask(
            &provider,
            &model,
            &root,
            route,
            request,
            timeout,
            self.temperature_depth > 0,
        )
    }
    fn lift(&mut self, values: Vec<V>, op: impl Fn(Vec<J>) -> Res<J> + 'static) -> Res<V> {
        let lazy = values.iter().any(|v| matches!(v, V::Maybe(_)));
        let recipe = move |rt: &mut Rt, s| {
            let mut confidence: f64 = 1.;
            let mut gate_confidence: f64 = 1.;
            let mut resolution = None;
            let mut resolution_gate_confidence = None;
            let mut reasons = vec![];
            let mut refs = vec![];
            let mut data = vec![];
            let mut generative = false;
            let mut decision = None;
            for v in &values {
                let v = rt.force(v, s)?;
                confidence = confidence.min(v.confidence);
                gate_confidence = gate_confidence.min(v.gate_confidence);
                if resolution.is_none() {
                    resolution = v.resolution.clone();
                    resolution_gate_confidence = v.resolution_gate_confidence;
                }
                generative |= v.generative;
                if decision.is_none() {
                    decision = v.decision.clone();
                }
                if v.generative && v.decision.is_none() && !v.because.is_empty() {
                    reasons.push(v.because);
                }
                refs = citations(&refs, &v.citations);
                data.push(v.data)
            }
            Ok(Sample {
                data: op(data)?,
                confidence,
                gate_confidence,
                resolution,
                resolution_gate_confidence,
                because: if generative && decision.is_none() {
                    format!("Derived from: {}", reasons.join(" | "))
                } else {
                    String::new()
                },
                citations: refs,
                generative,
                decision,
            })
        };
        if lazy {
            Ok(maybe(recipe, false))
        } else {
            Ok(V::Certain(recipe(self, 0)?))
        }
    }
    pub fn attribute(&mut self, scope: Scope, v: V, name: &str) -> Res<V> {
        if name == "owner" {
            return Ok(definite(match &v {
                V::Certain(_) => json!("certain"),
                V::Maybe(m) => json!(m.borrow().owner),
            }));
        }
        if [
            "because",
            "confidence",
            "citations",
            "decision",
            "resolution",
        ]
        .contains(&name)
        {
            let v = self.force(&v, scope)?;
            let data = match name {
                "because" => {
                    if !v.generative || v.decision.is_some() {
                        return fail(
                            "ExplainabilityError",
                            "because requires generative evidence without decision evidence",
                        );
                    }
                    json!(v.because)
                }
                "decision" => match v.decision.clone() {
                    Some(evidence) => evidence,
                    None => return fail("TypeError", "value has no decision evidence"),
                },
                "resolution" => {
                    let trace = v
                        .resolution
                        .clone()
                        .ok_or_else(|| type_fault("value has no resolution evidence"))?;
                    let mut metadata = v.clone();
                    // Exposing history exposes the actual first decision's data.
                    metadata.decision = Some(trace["decision"].clone());
                    metadata.gate_confidence = v
                        .gate_confidence
                        .min(v.resolution_gate_confidence.unwrap_or(v.gate_confidence));
                    metadata.because.clear();
                    metadata.data = trace;
                    return Ok(V::Certain(metadata));
                }
                "confidence" => json!(v.confidence),
                _ => json!(v.citations),
            };
            let mut metadata = Sample::new(data);
            if name == "decision" {
                metadata.decision = v.decision;
                metadata.confidence = v.confidence;
                metadata.gate_confidence = v.gate_confidence;
                metadata.resolution = v.resolution;
                metadata.resolution_gate_confidence = v.resolution_gate_confidence;
                metadata.citations = v.citations;
            }
            return Ok(V::Certain(metadata));
        }
        let n = name.to_owned();
        self.lift(vec![v], move |data| {
            let value = &data[0];
            match n.as_str() {
                "size" if value.is_array() => return Ok(json!(value.as_array().unwrap().len())),
                "size" if value.is_object() => return Ok(json!(value.as_object().unwrap().len())),
                "length" if value.is_string() => {
                    return Ok(json!(value.as_str().unwrap().chars().count()))
                }
                "keys" if value.is_object() => {
                    return Ok(json!(value.as_object().unwrap().keys().collect::<Vec<_>>()))
                }
                "values" if value.is_object() => {
                    return Ok(json!(value
                        .as_object()
                        .unwrap()
                        .values()
                        .collect::<Vec<_>>()))
                }
                _ => (),
            }
            value
                .as_object()
                .and_then(|m| m.get(&n))
                .cloned()
                .ok_or_else(|| Fault {
                    kind: "TypeError".into(),
                    message: format!("no field '{n}'"),
                    line: None,
                })
        })
    }
    pub fn unary(&mut self, op: &str, v: V) -> Res<V> {
        let op = op.to_owned();
        self.lift(vec![v], move |data| {
            let x = &data[0];
            match op.as_str() {
                "!" => match x.as_bool() {
                    Some(b) => Ok(json!(!b)),
                    None => fail("TypeError", "! requires Bool"),
                },
                _ => {
                    if let Some(n) = x.as_i64() {
                        return n.checked_neg().map(|x| json!(x)).ok_or_else(overflow);
                    }
                    if let Some(n) = x.as_f64() {
                        return number(-n);
                    }
                    fail("TypeError", "unary - requires a number")
                }
            }
        })
    }
    pub fn binary(&mut self, op: &str, a: V, b: V, line: usize) -> Res<V> {
        if ["==", "!="].contains(&op) && [&a, &b].iter().any(|v| matches!(v, V::Maybe(_))) {
            return fail(
                "TypeError",
                "== and != are undefined on maybes; use ~= or collapse with !!",
            );
        }
        if op == "to" {
            return self.lift(vec![a, b], |data| {
                Ok(json!({"first":data[0], "second":data[1]}))
            });
        }
        if op == "~=" || op == "?!=" {
            let decision_mode = op == "?!=";
            return Ok(maybe(
                move |rt, s| {
                    let a = rt.force(&a, s)?;
                    let b = rt.force(&b, s)?;
                    let mut answer = if decision_mode {
                        rt.ask_decision(
                            "difference",
                            json!([a.data, b.data]),
                            "Bool",
                            line,
                            json!([false, true]),
                        )?
                    } else {
                        rt.ask("equivalence", json!([a.data, b.data]), "Bool", line)?
                    };
                    answer.generative |= a.generative || b.generative;
                    if answer.decision.is_none() {
                        answer.decision = a.decision.or(b.decision);
                    }
                    answer.confidence = answer.confidence.min(a.confidence).min(b.confidence);
                    answer.gate_confidence = answer
                        .gate_confidence
                        .min(a.gate_confidence)
                        .min(b.gate_confidence);
                    answer.resolution_gate_confidence = if a.resolution.is_some() {
                        a.resolution_gate_confidence
                    } else {
                        b.resolution_gate_confidence
                    };
                    answer.resolution = a.resolution.or(b.resolution);
                    answer.citations =
                        citations(&citations(&answer.citations, &a.citations), &b.citations);
                    Ok(answer)
                },
                false,
            ));
        }
        let op = op.to_owned();
        self.lift(vec![a, b], move |data| operate(&op, &data[0], &data[1]))
    }
    pub fn call(&mut self, scope: Scope, name: &str, values: Vec<V>) -> Res<V> {
        match name {
            "sample" => {
                if values.len() != 1 {
                    return fail("TypeError", "sample(value) takes exactly one argument");
                }
                return Ok(V::Certain(self.force(&values[0], scope)?));
            }
            "hi" => {
                if !values.is_empty() {
                    return fail("TypeError", "hi() takes no arguments");
                }
                return Ok(V::Certain(Sample::explained(
                    json!("hi! 🐈‍⬛"),
                    "The standard library's sole certainty.",
                )));
            }
            "print" => {
                let mut parts = vec![];
                for v in values {
                    parts.push(display(&self.force(&v, scope)?.data))
                }
                println!("{}", parts.join(" "));
                return Ok(V::Certain(Sample::explained(
                    J::Null,
                    "The print intrinsic completed.",
                )));
            }
            "Human" => {
                if values.len() != 1 {
                    return fail("TypeError", "Human(lastMessage) takes one argument");
                }
                return self.lift(values, |data| intrinsic("Human", data));
            }
            _ => (),
        }
        if is_intrinsic(name) {
            let n = name.to_owned();
            return self.lift(values, move |data| intrinsic(&n, data));
        }
        let f = match self.lookup(scope, name)? {
            Binding::Function(f) => f,
            _ => return fail("TypeError", format!("'{name}' is not callable")),
        };
        if values.len() != f.params.len() {
            return fail(
                "TypeError",
                format!(
                    "{name} expects {} arguments, got {}",
                    f.params.len(),
                    values.len()
                ),
            );
        }
        let child = self.child(f.closure, name);
        for ((name, typ), v) in f.params.iter().zip(values) {
            let v = self.checked(v, typ)?;
            self.define(child, name, Binding::Value(v))?;
        }
        let result = (f.body)(self, child)?.unwrap_or_else(|| definite(J::Null));
        self.checked(result, &f.typ)
    }
    fn ask(&mut self, kind: &str, input: J, typ: &str, line: usize) -> Res<Sample> {
        if self.calls >= self.max_calls {
            return fail(
                "OracleError",
                format!("oracle call budget exhausted ({})", self.max_calls),
            );
        }
        self.calls += 1;
        let temperature = if self.temperature_depth == 0 && self.generative_route.is_none() {
            self.fallback_profile_temperature
                .unwrap_or(self.temperature)
        } else {
            self.temperature
        };
        let answer = if let Some(route) = self.generative_route.clone() {
            let request = json!({"protocol":"purrhaps/1","kind":kind,"input":input,"type":if typ.is_empty(){J::Null}else{json!(typ)},"line":line,"source":self.source,"temperature":temperature});
            self.configured(&route, request)?
        } else if self.command.is_empty() {
            self.demo(kind, input, typ, line)
        } else {
            let request = json!({"protocol":"purrhaps/1","kind":kind,"input":input,"type":if typ.is_empty(){J::Null}else{json!(typ)},"line":line,"source":self.source,"temperature":temperature});
            external(&self.command, self.timeout, &request)?
        };
        typed(validate(answer, self.source.lines().count(), None)?, typ)
    }
    fn ask_resolve(
        &mut self,
        input: J,
        typ: &str,
        line: usize,
        options: J,
        prior: Option<J>,
    ) -> Res<Sample> {
        if self.calls >= self.max_calls {
            return fail(
                "OracleError",
                format!("oracle call budget exhausted ({})", self.max_calls),
            );
        }
        self.calls += 1;
        let temperature = if self.temperature_depth == 0 && self.generative_route.is_none() {
            self.fallback_profile_temperature
                .unwrap_or(self.temperature)
        } else {
            self.temperature
        };
        let mut request = json!({"protocol":"purrhaps/1","kind":"resolve","input":input,"options":options,"type":if typ.is_empty(){J::Null}else{json!(typ)},"line":line,"source":self.source,"temperature":temperature});
        if let Some(prior) = prior {
            request["prior_decision"] = prior;
        }
        let answer = if let Some(route) = self.generative_route.clone() {
            self.configured(&route, request)?
        } else if self.command.is_empty() {
            self.demo("resolve", input, typ, line)
        } else {
            external(&self.command, self.timeout, &request)?
        };
        if !options.as_array().unwrap().iter().any(|v| {
            answer
                .get("value")
                .is_some_and(|data| strict_equal(v, data))
        }) {
            return fail(
                "OracleError",
                "resolve value must strictly match one supplied option",
            );
        }
        typed(validate(answer, self.source.lines().count(), None)?, typ)
    }
    pub fn list(&mut self, values: Vec<V>) -> Res<V> {
        self.lift(values, |data| Ok(J::Array(data)))
    }
    pub fn index(&mut self, receiver: V, index: V) -> Res<V> {
        self.lift(vec![receiver, index], |data| {
            indexed(&data[0], &data[1], false)
        })
    }
    pub fn method(&mut self, receiver: V, name: &str, mut values: Vec<V>) -> Res<V> {
        let n = name.to_owned();
        values.insert(0, receiver);
        self.lift(values, move |data| method(&n, &data[0], &data[1..]))
    }
    fn ask_decision(
        &mut self,
        kind: &str,
        input: J,
        typ: &str,
        line: usize,
        options: J,
    ) -> Res<Sample> {
        if self.calls >= self.max_calls {
            return fail(
                "OracleError",
                format!("oracle call budget exhausted ({})", self.max_calls),
            );
        }
        self.calls += 1;
        let answer = if let Some(route) = self.decision_route.clone() {
            let request = json!({"protocol":"purrhaps/2","kind":kind,"input":input,"type":if typ.is_empty(){J::Null}else{json!(typ)},"line":line,"source":self.source,"temperature":self.temperature,"options":options});
            self.configured(&route, request)?
        } else if self.decision_command.is_empty() {
            let value = if kind == "difference" {
                let eq = self.demo("equivalence", input.clone(), "Bool", line);
                json!(!eq["value"].as_bool().unwrap())
            } else {
                options[0].clone()
            };
            json!({"value":value, "confidence":if kind == "difference" {0.94} else {0.92}})
        } else {
            let request = json!({"protocol":"purrhaps/2", "kind":kind, "input":input, "type":if typ.is_empty(){J::Null}else{json!(typ)}, "line":line, "source":self.source, "temperature":self.temperature, "options":options});
            external(&self.decision_command, self.timeout, &request)?
        };
        let mut sample = validate(answer, self.source.lines().count(), Some(&options))?;
        let scores = sample.decision.take().unwrap_or(J::Null);
        sample.decision = Some(
            json!({"kind":kind, "input":input, "options":options, "selected":sample.data, "scores":scores}),
        );
        typed(sample, typ)
    }
    fn demo(&mut self, kind: &str, input: J, typ: &str, line: usize) -> J {
        if kind == "resolve" {
            return json!({"value":input["options"][0],"confidence":0.92,"because":"The offline demo selected the first permitted option; no model inference was performed.","citations":[],"caveats":[]});
        }
        if kind == "equivalence" {
            let normalize = |v: &J| {
                display(v)
                    .to_lowercase()
                    .trim()
                    .trim_matches(['.', '!', '?'])
                    .to_owned()
            };
            let a = normalize(&input[0]);
            let b = normalize(&input[1]);
            let groups = [
                vec!["sunny", "happy", "cheerful", "☀️"],
                vec!["cat", "kitty", "feline"],
            ];
            return json!({"value":a==b||groups.iter().any(|g|g.contains(&a.as_str())&&g.contains(&b.as_str())),"confidence":0.94,"because":"The demo oracle compared normalized words and its tiny synonym table.","citations":[line]});
        }
        let mut value = input.clone();
        let mut confidence = 0.92;
        let mut because = "The demo oracle preserved the declaration's input.";
        if input == json!({"self":"purrhaps"}) {
            value = json!("I can't introspect reliably");
            confidence = 0.26;
            because = "The demo oracle has no introspective machinery."
        } else if typ == "Int" && input.is_string() {
            let text = input.as_str().unwrap();
            let lower = text.to_lowercase();
            let n = if lower.contains("answer to everything") || lower.contains("meaning of life") {
                because = "The declaration strongly implied 42.";
                Some(42)
            } else {
                because = "The demo oracle extracted the first integer.";
                first_integer(text)
            };
            let mut n = n.unwrap_or_else(|| {
                confidence = 0.4;
                because = "The demo oracle could not infer an integer.";
                0
            });
            if self.temperature > 0. && confidence >= 0.8 {
                self.rng = self
                    .rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let delta = [0, 0, 0, -1, 1][((self.rng >> 32) % 5) as usize];
                n = n.saturating_add(delta)
            }
            value = json!(n);
        } else if typ == "String" {
            value = json!(display(&input))
        } else if typ == "Float" && input.is_number() {
            value = json!(input.as_f64().unwrap())
        }
        json!({"value":value,"confidence":confidence,"because":because,"citations":[line]})
    }
}
fn type_fault(message: impl Into<String>) -> Fault {
    Fault {
        kind: "TypeError".into(),
        message: message.into(),
        line: None,
    }
}
fn arity(name: &str, args: &[J], min: usize, max: usize) -> Res<()> {
    if args.len() < min || args.len() > max {
        return fail(
            "TypeError",
            format!("{name} expects {min}..{max} arguments, got {}", args.len()),
        );
    }
    Ok(())
}
fn count(value: &J) -> Res<usize> {
    let n = value
        .as_i64()
        .filter(|n| *n >= 0)
        .ok_or_else(|| type_fault("index/count must be a nonnegative Int"))?;
    usize::try_from(n).map_err(|_| type_fault("index/count is outside native range"))
}
fn indexed(receiver: &J, index: &J, nullable: bool) -> Res<J> {
    if let Some(map) = receiver.as_object() {
        let key = index
            .as_str()
            .ok_or_else(|| type_fault("map index must be a String"))?;
        return Ok(map.get(key).cloned().unwrap_or(J::Null));
    }
    if nullable && index.as_i64().is_some_and(|n| n < 0) {
        return Ok(J::Null);
    }
    let n = count(index)?;
    let result = match receiver {
        J::Array(items) => items.get(n).cloned(),
        J::String(text) => text.chars().nth(n).map(|c| json!(c.to_string())),
        _ => return fail("TypeError", "indexing requires a list, map, or string"),
    };
    match result {
        Some(value) => Ok(value),
        None if nullable => Ok(J::Null),
        None => fail("TypeError", "index out of bounds"),
    }
}
fn is_intrinsic(name: &str) -> bool {
    [
        "listOf",
        "emptyList",
        "mapOf",
        "emptyMap",
        "Pair",
        "Message",
        "Prompt",
        "DecisionInput",
        "TokenUsage",
        "ModelResponse",
        "Human",
        "abs",
        "minOf",
        "maxOf",
    ]
    .contains(&name)
}
fn intrinsic(name: &str, args: Vec<J>) -> Res<J> {
    match name {
        "listOf" => Ok(J::Array(args)),
        "emptyList" => {
            arity(name, &args, 0, 0)?;
            Ok(json!([]))
        }
        "emptyMap" => {
            arity(name, &args, 0, 0)?;
            Ok(json!({}))
        }
        "Pair" => {
            arity(name, &args, 2, 2)?;
            Ok(json!({"first":args[0],"second":args[1]}))
        }
        "mapOf" => {
            let mut map = serde_json::Map::new();
            for pair in args {
                if !shape(&pair, "Pair")? {
                    return fail("TypeError", "mapOf requires Pair values with String keys");
                }
                let key = pair["first"]
                    .as_str()
                    .ok_or_else(|| type_fault("mapOf keys must be String"))?;
                map.insert(key.into(), pair["second"].clone());
            }
            Ok(J::Object(map))
        }
        "Human" | "Message" | "Prompt" | "DecisionInput" | "TokenUsage" | "ModelResponse" => {
            let fields: &[&str] = match name {
                "Human" => &["lastMessage"],
                "Message" => &["role", "content"],
                "Prompt" => &["messages"],
                "DecisionInput" => &["question", "options", "context"],
                "TokenUsage" => &["inputTokens", "outputTokens"],
                _ => &["text", "finishReason", "usage"],
            };
            arity(name, &args, fields.len(), fields.len())?;
            let result = J::Object(
                fields
                    .iter()
                    .zip(args)
                    .map(|(k, v)| ((*k).into(), v))
                    .collect(),
            );
            if !shape(&result, name)? {
                return fail("TypeError", format!("invalid {name} fields"));
            }
            Ok(result)
        }
        "abs" => {
            arity(name, &args, 1, 1)?;
            absolute(&args[0])
        }
        "minOf" | "maxOf" => {
            arity(name, &args, 2, usize::MAX)?;
            if args.iter().any(|v| !v.is_number()) {
                return fail("TypeError", "minOf/maxOf require numbers");
            }
            let mut best = &args[0];
            for value in &args[1..] {
                let op = if name == "minOf" { "<" } else { ">" };
                if operate(op, value, best)?.as_bool() == Some(true) {
                    best = value;
                }
            }
            Ok(best.clone())
        }
        _ => fail("TypeError", format!("unknown intrinsic {name}")),
    }
}
fn absolute(value: &J) -> Res<J> {
    if let Some(n) = value.as_i64() {
        return n.checked_abs().map(|n| json!(n)).ok_or_else(overflow);
    }
    if let Some(n) = value.as_f64() {
        return number(n.abs());
    }
    fail("TypeError", "abs requires a number")
}
fn numeric_text(text: &str, decimal: bool) -> bool {
    let bytes = text.as_bytes();
    let mut i = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let start = i;
    while bytes.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
    }
    let mut digits = i - start;
    if decimal && bytes.get(i) == Some(&b'.') {
        i += 1;
        let start = i;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        digits += i - start;
    }
    if digits == 0 {
        return false;
    }
    if decimal && matches!(bytes.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let start = i;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        if start == i {
            return false;
        }
    }
    i == bytes.len()
}
fn method(name: &str, receiver: &J, args: &[J]) -> Res<J> {
    if name == "toString" {
        arity(name, args, 0, 0)?;
        return Ok(json!(display(receiver)));
    }
    if name == "abs" {
        arity(name, args, 0, 0)?;
        return absolute(receiver);
    }
    if name == "coerceIn" {
        arity(name, args, 2, 2)?;
        if !receiver.is_number() || args.iter().any(|v| !v.is_number()) {
            return fail("TypeError", "coerceIn requires numbers");
        }
        if operate(">", &args[0], &args[1])?.as_bool() == Some(true) {
            return fail("TypeError", "coerceIn minimum exceeds maximum");
        }
        return Ok(
            if operate("<", receiver, &args[0])?.as_bool() == Some(true) {
                args[0].clone()
            } else if operate(">", receiver, &args[1])?.as_bool() == Some(true) {
                args[1].clone()
            } else {
                receiver.clone()
            },
        );
    }
    if ["isEmpty", "isNotEmpty", "contains"].contains(&name) {
        arity(
            name,
            args,
            usize::from(name == "contains"),
            usize::from(name == "contains"),
        )?;
        let result = match receiver {
            J::Array(items) => {
                if name == "contains" {
                    items.iter().any(|v| strict_equal(v, &args[0]))
                } else {
                    items.is_empty()
                }
            }
            J::Object(map) => {
                if name == "contains" {
                    let key = args[0]
                        .as_str()
                        .ok_or_else(|| type_fault("map contains requires a String key"))?;
                    map.contains_key(key)
                } else {
                    map.is_empty()
                }
            }
            J::String(text) => {
                if name == "contains" {
                    let part = args[0]
                        .as_str()
                        .ok_or_else(|| type_fault("string contains requires String"))?;
                    text.contains(part)
                } else {
                    text.is_empty()
                }
            }
            _ => {
                return fail(
                    "TypeError",
                    format!("{name} requires a list, map, or string"),
                )
            }
        };
        return Ok(json!(if name == "isNotEmpty" {
            !result
        } else {
            result
        }));
    }
    if ["first", "last", "get", "getOrNull"].contains(&name) {
        if !receiver.is_array() && !receiver.is_string() {
            return fail("TypeError", format!("{name} requires a list or string"));
        }
        if name == "get" || name == "getOrNull" {
            arity(name, args, 1, 1)?;
            return indexed(receiver, &args[0], name == "getOrNull");
        }
        arity(name, args, 0, 0)?;
        let len = match receiver {
            J::Array(items) => items.len(),
            J::String(text) => text.chars().count(),
            _ => unreachable!(),
        };
        if len == 0 {
            return fail(
                "TypeError",
                format!("{name} requires a nonempty collection"),
            );
        }
        return indexed(
            receiver,
            &json!(if name == "first" { 0 } else { len - 1 }),
            false,
        );
    }
    if let J::Array(items) = receiver {
        return match name {
            "joinToString" => {
                arity(name, args, 0, 1)?;
                let sep = if args.is_empty() {
                    ", "
                } else {
                    args[0]
                        .as_str()
                        .ok_or_else(|| type_fault("joinToString separator must be String"))?
                };
                Ok(json!(items
                    .iter()
                    .map(display)
                    .collect::<Vec<_>>()
                    .join(sep)))
            }
            "distinct" => {
                arity(name, args, 0, 0)?;
                let mut result = vec![];
                for item in items {
                    if !result.iter().any(|v| strict_equal(item, v)) {
                        result.push(item.clone());
                    }
                }
                Ok(J::Array(result))
            }
            "reversed" => {
                arity(name, args, 0, 0)?;
                Ok(J::Array(items.iter().rev().cloned().collect()))
            }
            "take" | "drop" => {
                arity(name, args, 1, 1)?;
                let n = count(&args[0])?.min(items.len());
                Ok(json!(if name == "take" {
                    &items[..n]
                } else {
                    &items[n..]
                }))
            }
            "sorted" => {
                arity(name, args, 0, 0)?;
                let numeric = items.iter().all(J::is_number);
                let strings = items.iter().all(J::is_string);
                if !numeric && !strings {
                    return fail("TypeError", "sorted requires only numbers or only strings");
                }
                let mut result = items.clone();
                result.sort_by(|a, b| {
                    if let (Some(a), Some(b)) = (a.as_i64(), b.as_i64()) {
                        a.cmp(&b)
                    } else if numeric {
                        a.as_f64()
                            .unwrap()
                            .partial_cmp(&b.as_f64().unwrap())
                            .unwrap()
                    } else {
                        a.as_str().unwrap().cmp(b.as_str().unwrap())
                    }
                });
                Ok(J::Array(result))
            }
            "sum" => {
                arity(name, args, 0, 0)?;
                if items.iter().any(|v| !v.is_number()) {
                    return fail("TypeError", "sum requires only numbers");
                }
                let mut result = json!(0);
                for item in items {
                    result = operate("+", &result, item)?;
                }
                Ok(result)
            }
            _ => fail("TypeError", format!("no list method '{name}'")),
        };
    }
    if let J::String(text) = receiver {
        return match name {
            "trim" | "lowercase" | "uppercase" => {
                arity(name, args, 0, 0)?;
                Ok(json!(match name {
                    "trim" => trim(text).to_owned(),
                    "lowercase" => text.to_lowercase(),
                    _ => text.to_uppercase(),
                }))
            }
            "startsWith" | "endsWith" | "split" => {
                arity(name, args, 1, 1)?;
                let part = args[0]
                    .as_str()
                    .ok_or_else(|| type_fault(format!("{name} requires String")))?;
                match name {
                    "startsWith" => Ok(json!(text.starts_with(part))),
                    "endsWith" => Ok(json!(text.ends_with(part))),
                    _ => {
                        if part.is_empty() {
                            return fail("TypeError", "split separator cannot be empty");
                        }
                        Ok(json!(text.split(part).collect::<Vec<_>>()))
                    }
                }
            }
            "replace" => {
                arity(name, args, 2, 2)?;
                let old = args[0]
                    .as_str()
                    .ok_or_else(|| type_fault("replace requires String arguments"))?;
                let new = args[1]
                    .as_str()
                    .ok_or_else(|| type_fault("replace requires String arguments"))?;
                Ok(json!(text.replace(old, new)))
            }
            "substring" => {
                arity(name, args, 1, 2)?;
                let start = count(&args[0])?;
                let chars = text.chars().collect::<Vec<_>>();
                let end = if args.len() == 2 {
                    count(&args[1])?
                } else {
                    chars.len()
                };
                if start > end || end > chars.len() {
                    return fail("TypeError", "substring bounds are invalid");
                }
                Ok(json!(chars[start..end].iter().collect::<String>()))
            }
            "toIntOrNull" => {
                arity(name, args, 0, 0)?;
                let value = trim(text);
                Ok(if numeric_text(value, false) {
                    value
                        .parse::<i64>()
                        .ok()
                        .map(|n| json!(n))
                        .unwrap_or(J::Null)
                } else {
                    J::Null
                })
            }
            "toDoubleOrNull" => {
                arity(name, args, 0, 0)?;
                let value = trim(text);
                Ok(if numeric_text(value, true) {
                    value
                        .parse::<f64>()
                        .ok()
                        .filter(|n| n.is_finite())
                        .map(|n| json!(n))
                        .unwrap_or(J::Null)
                } else {
                    J::Null
                })
            }
            _ => fail("TypeError", format!("no string method '{name}'")),
        };
    }
    fail(
        "TypeError",
        format!("no method '{name}' for {}", typename(receiver)),
    )
}

fn first_integer(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    for i in 0..b.len() {
        if b[i].is_ascii_digit() || (b[i] == b'-' && b.get(i + 1).is_some_and(u8::is_ascii_digit)) {
            let mut end = i + 1;
            while end < b.len() && b[end].is_ascii_digit() {
                end += 1
            }
            return s[i..end].parse().ok();
        }
    }
    None
}
fn overflow() -> Fault {
    Fault {
        kind: "TypeError".into(),
        message: "integer overflow (Rust backend Int is signed 64-bit)".into(),
        line: None,
    }
}
fn number(n: f64) -> Res<J> {
    if n.is_finite() {
        Ok(json!(n))
    } else {
        fail("TypeError", "non-finite arithmetic result")
    }
}
// Ordinary equality retains numeric coercion; bounded choices use strict_equal.
fn equal(a: &J, b: &J) -> bool {
    let numeric = |v: &J| {
        v.as_f64()
            .or_else(|| v.as_bool().map(|b| if b { 1. } else { 0. }))
    };
    if let (Some(a), Some(b)) = (a.as_i64(), b.as_i64()) {
        return a == b;
    }
    if let (Some(a), Some(b)) = (numeric(a), numeric(b)) {
        return a == b;
    }
    match (a, b) {
        (J::Array(a), J::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| equal(a, b))
        }
        (J::Object(a), J::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, a)| b.get(key).is_some_and(|b| equal(a, b)))
        }
        _ => a == b,
    }
}
fn operate(op: &str, a: &J, b: &J) -> Res<J> {
    if op == "==" || op == "!=" {
        return Ok(json!(equal(a, b) == (op == "==")));
    }
    if op == "&&" || op == "||" {
        return match (a.as_bool(), b.as_bool()) {
            (Some(a), Some(b)) => Ok(json!(if op == "&&" { a && b } else { a || b })),
            _ => fail("TypeError", format!("{op} requires Bool operands")),
        };
    }
    if let (Some(a), Some(b)) = (a.as_str(), b.as_str()) {
        return match op {
            "+" => Ok(json!(format!("{a}{b}"))),
            "<" => Ok(json!(a < b)),
            ">" => Ok(json!(a > b)),
            "<=" => Ok(json!(a <= b)),
            ">=" => Ok(json!(a >= b)),
            _ => fail(
                "TypeError",
                format!("{op} requires numbers (or two strings for +)"),
            ),
        };
    }
    if let (Some(x), Some(y)) = (a.as_i64(), b.as_i64()) {
        let integer = match op {
            "+" => Some(x.checked_add(y)),
            "-" => Some(x.checked_sub(y)),
            "*" => Some(x.checked_mul(y)),
            "%" => {
                if y == 0 {
                    return fail("TypeError", "invalid operation: integer modulo by zero");
                }
                let r = x.checked_rem(y).unwrap_or(0);
                Some(if r != 0 && (r < 0) != (y < 0) {
                    r.checked_add(y)
                } else {
                    Some(r)
                })
            }
            "<" => return Ok(json!(x < y)),
            ">" => return Ok(json!(x > y)),
            "<=" => return Ok(json!(x <= y)),
            ">=" => return Ok(json!(x >= y)),
            _ => None,
        };
        if let Some(result) = integer {
            return result.map(|v| json!(v)).ok_or_else(overflow);
        }
    }
    if let (Some(x), Some(y)) = (a.as_f64(), b.as_f64()) {
        return match op {
            "+" => number(x + y),
            "-" => number(x - y),
            "*" => number(x * y),
            "/" => {
                if y == 0. {
                    fail("TypeError", "invalid operation: division by zero")
                } else {
                    number(x / y)
                }
            }
            "%" => {
                if y == 0. {
                    fail("TypeError", "invalid operation: float modulo")
                } else {
                    let r = x % y;
                    number(if r != 0. && (r < 0.) != (y < 0.) {
                        r + y
                    } else {
                        r
                    })
                }
            }
            "<" => Ok(json!(x < y)),
            ">" => Ok(json!(x > y)),
            "<=" => Ok(json!(x <= y)),
            ">=" => Ok(json!(x >= y)),
            _ => fail("TypeError", "invalid binary operator"),
        };
    }
    fail("TypeError", format!("invalid operands for {op}"))
}
fn validate(answer: J, lines: usize, options: Option<&J>) -> Res<Sample> {
    let bad = || Fault {
        kind: "OracleError".into(),
        message: "oracle command returned an invalid answer object".into(),
        line: None,
    };
    let obj = answer.as_object().ok_or_else(bad)?;
    if obj.keys().any(|k| {
        ![
            "value",
            "confidence",
            "because",
            "citations",
            "caveats",
            "scores",
        ]
        .contains(&k.as_str())
    }) {
        return Err(bad());
    }
    let data = obj.get("value").ok_or_else(bad)?.clone();
    let confidence = obj
        .get("confidence")
        .and_then(J::as_f64)
        .filter(|n| n.is_finite() && (0.0..=1.0).contains(n))
        .ok_or_else(bad)?;
    if let Some(options) = options {
        let options = options.as_array().ok_or_else(bad)?;
        if !options.iter().any(|option| strict_equal(option, &data)) {
            return fail(
                "OracleError",
                "decision value must strictly match one supplied option",
            );
        }
        let scores = obj.get("scores").cloned().unwrap_or(J::Null);
        if !scores.is_null()
            && !scores.as_array().is_some_and(|scores| {
                scores.len() == options.len()
                    && scores.iter().all(|s| {
                        s.as_f64()
                            .is_some_and(|n| n.is_finite() && (0.0..=1.0).contains(&n))
                    })
            })
        {
            return fail("OracleError", "decision scores must align with options and contain finite numbers between 0 and 1");
        }
        return Ok(Sample {
            data,
            confidence,
            gate_confidence: confidence,
            resolution: None,
            resolution_gate_confidence: None,
            because: String::new(),
            citations: vec![],
            generative: false,
            decision: Some(scores),
        });
    }
    let because = obj
        .get("because")
        .and_then(J::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(bad)?
        .to_owned();
    let mut refs = vec![];
    if let Some(c) = obj.get("citations") {
        for n in c.as_array().ok_or_else(bad)? {
            let n = n.as_i64().ok_or_else(bad)?;
            if n < 1 || n as usize > lines {
                return fail(
                    "ConfabulationError",
                    format!("oracle cited nonexistent line {n}"),
                );
            }
            refs.push(n as usize)
        }
    }
    let caveats = if let Some(c) = obj.get("caveats") {
        let c = c.as_array().ok_or_else(bad)?;
        if c.iter().any(|v| !v.is_string()) {
            return Err(bad());
        }
        c.len()
    } else {
        0
    };
    let lower = because.to_lowercase();
    let words: Vec<_> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .collect();
    // Match the Python runtime's simple "line N" prose citation convention.
    for (i, _) in lower.match_indices("line") {
        if i > 0
            && lower[..i]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_')
        {
            continue;
        }
        let tail = &lower[i + 4..];
        if !tail.starts_with(char::is_whitespace) {
            continue;
        }
        let digits: String = tail
            .trim_start()
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        if let Ok(n) = digits.parse::<usize>() {
            if n < 1 || n > lines {
                return fail(
                    "ConfabulationError",
                    format!("oracle cited nonexistent line {n}"),
                );
            }
        }
    }
    let hedges = words
        .iter()
        .filter(|w| {
            [
                "perhaps",
                "maybe",
                "possibly",
                "probably",
                "however",
                "although",
                "uncertain",
            ]
            .contains(w)
        })
        .count();
    if caveats.max(hedges) > 3 {
        return fail(
            "HedgeOverflowError",
            "rationale contains more than three caveats; please meow decisively",
        );
    }
    Ok(Sample {
        data,
        confidence,
        gate_confidence: confidence,
        resolution: None,
        resolution_gate_confidence: None,
        because,
        citations: refs,
        generative: true,
        decision: None,
    })
}
fn external(command: &[String], timeout: Duration, request: &J) -> Res<J> {
    let output = providers::run(command, request.to_string(), None, timeout)?;
    serde_json::from_str(&output).map_err(|_| Fault {
        kind: "OracleError".into(),
        message: "oracle command returned an invalid answer object".into(),
        line: None,
    })
}

pub fn launch(source: &str, entry: Block) {
    let mut seed = 0;
    let mut max_calls = 1000;
    let mut max_calls_set = false;
    let mut timeout_set = false;
    let mut project_path = None;
    let mut no_project = false;
    let mut model_route = None;
    let mut decision_route = None;
    let mut check = false;
    let mut seconds = 30.;
    let mut stats = false;
    let mut command = vec![];
    let mut decision_command = vec![];
    let mut args = std::env::args().skip(1);
    let parse = (|| -> Result<(), String> {
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--stats" => stats = true,
                "--check" => check = true,
                "--project" => {
                    project_path = Some(args.next().ok_or("missing project path")?);
                }
                "--no-project" => no_project = true,
                "--model" => {
                    model_route = Some(args.next().ok_or("missing model route")?);
                }
                "--decision-model" => {
                    decision_route = Some(args.next().ok_or("missing decision route")?);
                }
                "--seed" => {
                    let input = args.next().ok_or("missing seed")?;
                    let digits = input.trim_start_matches(['-', '+']);
                    if digits.is_empty()
                        || !digits.chars().all(|c| c.is_ascii_digit())
                        || input.starts_with("-+")
                        || input.starts_with("+-")
                        || input.starts_with("--")
                        || input.starts_with("++")
                    {
                        return Err("seed must be an integer".into());
                    }
                    seed = digits.bytes().fold(0u64, |n, digit| {
                        n.wrapping_mul(10).wrapping_add((digit - b'0') as u64)
                    });
                    if input.starts_with('-') {
                        seed = seed.wrapping_neg();
                    }
                }
                "--max-calls" => {
                    max_calls_set = true;
                    max_calls = args
                        .next()
                        .ok_or("missing call budget")?
                        .parse::<usize>()
                        .map_err(|_| "invalid call budget")?
                }
                "--timeout" => {
                    timeout_set = true;
                    seconds = args
                        .next()
                        .ok_or("missing timeout")?
                        .parse::<f64>()
                        .map_err(|_| "invalid timeout")?
                }
                "--oracle" => {
                    if !command.is_empty() {
                        return Err("--oracle may be specified once, before --oracle-arg".into());
                    }
                    command.push(args.next().ok_or("missing oracle executable")?)
                }
                "--oracle-arg" => {
                    if command.is_empty() {
                        return Err("--oracle-arg requires --oracle first".into());
                    }
                    command.push(args.next().ok_or("missing oracle argument")?)
                }
                "--decision-oracle" => {
                    if !decision_command.is_empty() {
                        return Err(
                            "--decision-oracle may be specified once, before --decision-oracle-arg"
                                .into(),
                        );
                    }
                    decision_command.push(args.next().ok_or("missing decision oracle executable")?)
                }
                "--decision-oracle-arg" => {
                    if decision_command.is_empty() {
                        return Err("--decision-oracle-arg requires --decision-oracle first".into());
                    }
                    decision_command.push(args.next().ok_or("missing decision oracle argument")?)
                }
                "--help" => {
                    println!("Purrhaps native program: [--seed N] [--max-calls N] [--timeout SECONDS] [--stats] [--project FILE_OR_DIR | --no-project] [--model ROUTE] [--decision-model ROUTE] [--check] [--oracle EXECUTABLE --oracle-arg ARG ...] [--decision-oracle EXECUTABLE --decision-oracle-arg ARG ...]");
                    std::process::exit(0)
                }
                _ => return Err(format!("unknown option: {arg}")),
            }
        }
        if no_project && project_path.is_some() {
            return Err("--project conflicts with --no-project".into());
        }
        if model_route.is_some() && !command.is_empty()
            || decision_route.is_some() && !decision_command.is_empty()
        {
            return Err("model route conflicts with explicit oracle command".into());
        }
        if max_calls == 0 || !seconds.is_finite() || seconds <= 0. || seconds > 86400. {
            return Err("max-calls must be positive; timeout must be in (0, 86400]".into());
        }
        Ok(())
    })();
    if let Err(message) = parse {
        eprintln!("ArgumentError: {message}");
        std::process::exit(2)
    }
    let setup = (|| -> Res<Option<Project>> {
        let cwd = std::env::current_dir().map_err(|_| Fault {
            kind: "ConfigurationError".into(),
            message: "cannot discover project directory".into(),
            line: None,
        })?;
        let path = if no_project {
            None
        } else {
            project_path
                .map(std::path::PathBuf::from)
                .or_else(|| Project::discover(&cwd))
        };
        let project = path.map(|p| Project::load(&p)).transpose()?;
        if let Some(p) = &project {
            if !max_calls_set {
                max_calls = p.data["runtime"]["max_calls"]
                    .as_u64()
                    .map(|v| v as usize)
                    .unwrap_or(1000);
            }
            if !timeout_set {
                seconds = p.data["runtime"]["timeout"].as_f64().unwrap_or(30.);
            }
            for (route, lane) in [(&model_route, "generative"), (&decision_route, "decision")] {
                if let Some(route) = route {
                    p.route(route, Some(lane))?;
                }
            }
        } else if model_route.is_some() || decision_route.is_some() {
            return fail(
                "ConfigurationError",
                "model selection requires a configured project",
            );
        }
        Ok(project)
    })();
    let project = match setup {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}: {}", e.kind, e.message);
            std::process::exit(1)
        }
    };
    if check {
        println!("Checked native project configuration.");
        return;
    }
    let mut rt = Rt::new(
        source,
        seed,
        max_calls,
        command,
        Duration::from_secs_f64(seconds),
    );
    rt.decision_command = decision_command;
    rt.timeout_override = if timeout_set {
        Some(Duration::from_secs_f64(seconds))
    } else {
        None
    };
    if let Some(p) = project {
        let root_route = model_route
            .as_deref()
            .unwrap_or_else(|| p.data["defaults"]["generative"].as_str().unwrap());
        rt.fallback_profile_temperature = p
            .route(root_route, Some("generative"))
            .expect("validated default route")
            .1["temperature"]
            .as_f64();
        rt.generative_route = if rt.command.is_empty() {
            Some(
                model_route
                    .unwrap_or_else(|| p.data["defaults"]["generative"].as_str().unwrap().into()),
            )
        } else {
            None
        };
        rt.decision_route = if rt.decision_command.is_empty() {
            Some(
                decision_route
                    .unwrap_or_else(|| p.data["defaults"]["decision"].as_str().unwrap().into()),
            )
        } else {
            None
        };
        rt.project = Some(p);
    }
    let result = entry(&mut rt, 0).and_then(|v| {
        if v.is_some() {
            fail("PurrhapsError", "return is only valid inside a function")
        } else {
            Ok(())
        }
    });
    if stats {
        eprintln!(
            "oracle={} decision_oracle={} calls={}",
            if rt.generative_route.is_some() {
                "ConfiguredOracle"
            } else if rt.command.is_empty() {
                "DemoOracle"
            } else {
                "CommandOracle"
            },
            if rt.decision_route.is_some() {
                "ConfiguredOracle"
            } else if rt.decision_command.is_empty() {
                "DemoOracle"
            } else {
                "CommandOracle"
            },
            rt.calls
        )
    }
    if let Err(e) = result {
        match e.line {
            Some(line) => eprintln!("{}: line {}: {}", e.kind, line, e.message),
            None => eprintln!("{}: {}", e.kind, e.message),
        }
        std::process::exit(1)
    }
}

#[cfg(test)]
mod project_scope_tests {
    use super::*;
    fn failing(rt: &mut Rt, _scope: Scope) -> Res<Option<V>> {
        assert_eq!(rt.generative_route.as_deref(), Some("inner"));
        fail("OracleError", "fixture failure")
    }
    fn returning(rt: &mut Rt, _scope: Scope) -> Res<Option<V>> {
        assert_eq!(rt.decision_route.as_deref(), Some("decision_inner"));
        Ok(Some(definite(json!(7))))
    }
    fn temperature_failure(rt: &mut Rt, _scope: Scope) -> Res<Option<V>> {
        assert_eq!(rt.temperature, 0.2);
        assert_eq!(rt.temperature_depth, 1);
        fail("OracleError", "fixture failure")
    }
    fn escalation_seed(_rt: &mut Rt, _scope: Scope) -> Res<V> {
        Ok(literal(
            json!({"question":"pick","options":[1,2],"context":null}),
            1,
        ))
    }
    fn escalation_restores_routes_and_rng_on_errors_and_success() {
        for (max_calls, policy, expected_calls, expected_kind) in [
            (10, "error", 2, "UncertaintyException"),
            (1, "random", 1, "OracleError"),
            (10, "random", 2, ""),
        ] {
            let mut rt = Rt::new("fixture", 42, max_calls, vec![], Duration::from_secs(1));
            rt.project = Some(Project {
                root: std::path::PathBuf::from("."),
                data: json!({"providers":{"offline":{"transport":"demo"}},"models":{"g":{"kind":"generative","provider":"offline","temperature":0.1},"d":{"kind":"decision","provider":"offline"}},"routes":{"inner":"g","decision_inner":"d"},"escalation":{"generative":"inner","decision":"decision_inner","threshold":1.0,"on_low_confidence":policy}}),
            });
            rt.generative_route = Some("outer".into());
            rt.decision_route = Some("decision_outer".into());
            rt.escalate(0, "x", "Int", escalation_seed, 1).unwrap();
            assert_eq!(rt.calls, 0);
            let x = rt.name(0, "x").unwrap();
            let result = rt.force(&x, 0);
            assert_eq!(rt.calls, expected_calls);
            assert_eq!(rt.generative_route.as_deref(), Some("outer"));
            assert_eq!(rt.decision_route.as_deref(), Some("decision_outer"));
            assert_eq!(rt.temperature, 0.7);
            assert_eq!(rt.temperature_depth, 0);
            assert_eq!(rt.rng, 42);
            if expected_kind.is_empty() {
                let sample = result.unwrap();
                assert_eq!(sample.confidence, 0.0);
                assert_eq!(sample.gate_confidence, 1.0);
                assert_eq!(sample.resolution_gate_confidence, Some(0.92));
                assert_ne!(rt.escalation_rng, 42);
            } else {
                assert_eq!(result.err().unwrap().kind, expected_kind);
                assert_eq!(rt.escalation_rng, 42);
            }
        }
    }
    #[test]
    fn route_and_temperature_restore_after_errors_and_returns() {
        let mut rt = Rt::new("", 0, 10, vec![], Duration::from_secs(1));
        rt.project = Some(Project {
            root: std::path::PathBuf::from("."),
            data: json!({"providers":{"offline":{"transport":"demo"}},"models":{"g":{"kind":"generative","provider":"offline"},"d":{"kind":"decision","provider":"offline"}},"routes":{"inner":"g","decision_inner":"d"}}),
        });
        rt.generative_route = Some("outer".into());
        rt.decision_route = Some("decision_outer".into());
        assert!(rt.model(0, definite(json!("inner")), failing).is_err());
        assert_eq!(rt.generative_route.as_deref(), Some("outer"));
        assert!(rt
            .model(0, definite(json!("decision_inner")), returning)
            .unwrap()
            .is_some());
        assert_eq!(rt.decision_route.as_deref(), Some("decision_outer"));
        assert!(rt
            .temperature(0, definite(json!(0.2)), temperature_failure)
            .is_err());
        assert_eq!(rt.temperature, 0.7);
        assert_eq!(rt.temperature_depth, 0);
        assert_eq!(rt.calls, 0);
        escalation_restores_routes_and_rng_on_errors_and_success();
    }
}
