//! Runtime-only project configuration. Never resolves environment variables.
use super::*;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub(super) struct Project {
    pub root: PathBuf,
    pub data: J,
}
fn bad<T>() -> Res<T> {
    fail(
        "ConfigurationError",
        "invalid project manifest configuration",
    )
}
fn keys(v: &J, allowed: &[&str]) -> Res<()> {
    if !v
        .as_object()
        .is_some_and(|o| o.keys().all(|k| allowed.contains(&k.as_str())))
    {
        return bad();
    }
    Ok(())
}
fn text(v: &J) -> bool {
    v.as_str().is_some_and(|s| !s.trim().is_empty())
}
fn positive(v: &J) -> bool {
    v.as_f64()
        .is_some_and(|n| n.is_finite() && n > 0. && n <= 86400.)
}
fn argv(v: &J) -> bool {
    v.as_array().is_some_and(|a| {
        !a.is_empty()
            && text(&a[0])
            && a.iter()
                .all(|s| s.as_str().is_some_and(|s| !s.contains('\0')))
    })
}
fn credential(k: &str) -> bool {
    let s: String = k
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    [
        "apikey",
        "apikeyenv",
        "authorization",
        "proxyauthorization",
        "headers",
        "header",
        "httpheaders",
        "extraheaders",
        "defaultheaders",
        "credential",
        "credentials",
        "password",
        "secret",
        "clientsecret",
        "token",
        "accesstoken",
        "refreshtoken",
        "authtoken",
        "bearertoken",
        "auth",
        "authentication",
    ]
    .contains(&s.as_str())
        || [
            "apikey",
            "authorization",
            "password",
            "secret",
            "credential",
            "header",
        ]
        .iter()
        .any(|part| s.contains(part))
}
fn safe_options(v: &J) -> bool {
    match v {
        J::Object(o) => o.iter().all(|(k, v)| !credential(k) && safe_options(v)),
        J::Array(a) => a.iter().all(safe_options),
        _ => finite_json(v),
    }
}
fn json_value(v: toml::Value) -> Res<J> {
    Ok(match v {
        toml::Value::String(v) => json!(v),
        toml::Value::Integer(v) => json!(v),
        toml::Value::Float(v) if v.is_finite() => json!(v),
        toml::Value::Boolean(v) => json!(v),
        toml::Value::Array(v) => J::Array(v.into_iter().map(json_value).collect::<Res<_>>()?),
        toml::Value::Table(v) => J::Object(
            v.into_iter()
                .map(|(k, v)| Ok((k, json_value(v)?)))
                .collect::<Res<_>>()?,
        ),
        _ => return bad(),
    })
}
impl Project {
    pub fn discover(start: &Path) -> Option<PathBuf> {
        let mut path = start.to_path_buf();
        loop {
            let candidate = path.join("purrhaps.toml");
            if candidate.is_file() {
                return Some(candidate);
            }
            if !path.pop() {
                return None;
            }
        }
    }
    pub fn load(path: &Path) -> Res<Self> {
        let path = if path.is_dir() {
            path.join("purrhaps.toml")
        } else {
            path.to_path_buf()
        };
        let path = path.canonicalize().map_err(|_| Fault {
            kind: "ConfigurationError".into(),
            message: "cannot read project manifest".into(),
            line: None,
        })?;
        let source = std::fs::read_to_string(&path).map_err(|_| Fault {
            kind: "ConfigurationError".into(),
            message: "cannot read project manifest".into(),
            line: None,
        })?;
        let parsed = source.parse::<toml::Value>().map_err(|_| Fault {
            kind: "ConfigurationError".into(),
            message: "invalid TOML project manifest".into(),
            line: None,
        })?;
        let data = json_value(parsed)?;
        keys(
            &data,
            &[
                "manifest_version",
                "project",
                "runtime",
                "providers",
                "models",
                "routes",
                "defaults",
                "escalation",
            ],
        )?;
        if data["manifest_version"].as_i64() != Some(1) {
            return bad();
        }
        keys(&data["project"], &["name", "entry", "no_humans_allowed"])?;
        if !text(&data["project"]["name"]) {
            return bad();
        }
        if data["project"]
            .get("no_humans_allowed")
            .is_some_and(|v| !v.is_boolean())
        {
            return bad();
        }
        if let Some(entry) = data["project"].get("entry") {
            if !text(entry)
                || Path::new(entry.as_str().unwrap()).is_absolute()
                || (entry.as_str().unwrap().contains('\0')
                    || Path::new(entry.as_str().unwrap()) == Path::new("."))
            {
                return bad();
            }
        }
        if let Some(runtime) = data.get("runtime") {
            keys(runtime, &["max_calls", "timeout"])?;
            if runtime.get("max_calls").is_some_and(|n| {
                !n.as_u64()
                    .is_some_and(|n| n > 0 && usize::try_from(n).is_ok())
            }) || runtime.get("timeout").is_some_and(|n| !positive(n))
            {
                return bad();
            }
        }
        let providers = data["providers"].as_object().ok_or_else(|| Fault {
            kind: "ConfigurationError".into(),
            message: "providers must be a table".into(),
            line: None,
        })?;
        let models = data["models"].as_object().ok_or_else(|| Fault {
            kind: "ConfigurationError".into(),
            message: "models must be a table".into(),
            line: None,
        })?;
        for (name, p) in providers {
            if name.trim().is_empty() {
                return bad();
            }
            let transport = p["transport"].as_str().unwrap_or("");
            let allowed: &[&str] = match transport {
                "demo" => &["transport"],
                "command" | "codex" | "claude" => &["transport", "command"],
                "openai-chat" => &["transport", "base_url", "api_key_env", "response_format"],
                "ollama" | "ollama-decision" => &["transport", "base_url", "api_key_env"],
                _ => return bad(),
            };
            keys(p, allowed)?;
            if transport == "command" && !argv(&p["command"])
                || p.get("command").is_some_and(|v| !argv(v))
            {
                return bad();
            }
            if ["openai-chat", "ollama", "ollama-decision"].contains(&transport) {
                let url = p["base_url"].as_str().ok_or_else(|| Fault {
                    kind: "ConfigurationError".into(),
                    message: "HTTP provider needs base_url".into(),
                    line: None,
                })?;
                let parsed = url::Url::parse(url).map_err(|_| Fault {
                    kind: "ConfigurationError".into(),
                    message: "invalid HTTP base_url".into(),
                    line: None,
                })?;
                if !["http", "https"].contains(&parsed.scheme())
                    || parsed.host_str().is_none()
                    || !parsed.username().is_empty()
                    || parsed.password().is_some()
                    || parsed.query().is_some()
                    || parsed.fragment().is_some()
                    || url.chars().any(|c| c.is_control() || c.is_whitespace())
                    || (url.contains('\\')
                        || url.split("/").nth(2).is_some_and(|a| a.contains('@')))
                {
                    return bad();
                }
                if let Some(env) = p.get("api_key_env") {
                    let env = env.as_str().unwrap_or("");
                    if env.is_empty()
                        || !env.chars().enumerate().all(|(i, c)| {
                            c == '_' || c.is_ascii_alphabetic() || i > 0 && c.is_ascii_digit()
                        })
                    {
                        return bad();
                    }
                }
                if p.get("response_format").is_some_and(|v| {
                    ![Some("json_schema"), Some("json_object")].contains(&v.as_str())
                }) {
                    return bad();
                }
            }
        }
        for (name, m) in models {
            if name.trim().is_empty() {
                return bad();
            }
            keys(
                m,
                &[
                    "provider",
                    "kind",
                    "model",
                    "timeout",
                    "temperature",
                    "options",
                ],
            )?;
            let p = m["provider"]
                .as_str()
                .and_then(|n| providers.get(n))
                .ok_or_else(|| Fault {
                    kind: "ConfigurationError".into(),
                    message: "unknown model provider".into(),
                    line: None,
                })?;
            let t = p["transport"].as_str().unwrap();
            let kind = m["kind"].as_str().unwrap_or("");
            if !["generative", "decision"].contains(&kind)
                || m.get("model").is_some_and(|v| !text(v))
                || !["demo", "command"].contains(&t) && !text(&m["model"])
                || m.get("timeout").is_some_and(|v| !positive(v))
            {
                return bad();
            }
            if let Some(temp) = m.get("temperature") {
                if kind != "generative"
                    || ["codex", "claude", "ollama-decision"].contains(&t)
                    || !temp
                        .as_f64()
                        .is_some_and(|n| n.is_finite() && (0.0..=2.).contains(&n))
                {
                    return bad();
                }
            }
            if ["ollama", "codex", "claude"].contains(&t) && kind != "generative"
                || t == "ollama-decision" && kind != "decision"
            {
                return bad();
            }
            let empty = json!({});
            let opts = m.get("options").unwrap_or(&empty);
            if !opts.is_object() || !safe_options(opts) {
                return bad();
            }
            let opts = opts.as_object().unwrap();
            if ["demo", "ollama-decision"].contains(&t) && !opts.is_empty() {
                return bad();
            }
            for (k, v) in opts {
                if t == "openai-chat"
                    && [
                        "model",
                        "messages",
                        "stream",
                        "response_format",
                        "temperature",
                        "tools",
                        "tool_choice",
                    ]
                    .contains(&k.as_str())
                    || t == "ollama" && k == "temperature"
                {
                    return bad();
                }
                if t == "codex"
                    && (!["reasoning_effort", "service_tier"].contains(&k.as_str()) || !text(v))
                    || t == "claude" && (k != "effort" || !text(v))
                {
                    return bad();
                }
            }
        }
        let routes = data["routes"].as_object().ok_or_else(|| Fault {
            kind: "ConfigurationError".into(),
            message: "routes must be a table".into(),
            line: None,
        })?;
        for (name, m) in routes {
            if name.trim().is_empty() || !m.as_str().is_some_and(|m| models.contains_key(m)) {
                return bad();
            }
        }
        keys(&data["defaults"], &["generative", "decision"])?;
        let project = Self {
            root: path.parent().unwrap().into(),
            data,
        };
        for lane in ["generative", "decision"] {
            let route = project.data["defaults"][lane]
                .as_str()
                .ok_or_else(|| Fault {
                    kind: "ConfigurationError".into(),
                    message: "missing default route".into(),
                    line: None,
                })?;
            project.route(route, Some(lane))?;
        }
        if let Some(escalation) = project.data.get("escalation") {
            keys(
                escalation,
                &[
                    "decision",
                    "generative",
                    "threshold",
                    "exposure",
                    "on_low_confidence",
                ],
            )?;
            for lane in ["decision", "generative"] {
                if let Some(route) = escalation.get(lane) {
                    let Some(route) = route.as_str().filter(|s| !s.trim().is_empty()) else {
                        return bad();
                    };
                    project.route(route, Some(lane))?;
                }
            }
            if escalation.get("threshold").is_some_and(|v| {
                !v.as_f64()
                    .is_some_and(|n| n.is_finite() && (0.0..=1.0).contains(&n))
            }) || escalation
                .get("exposure")
                .is_some_and(|v| ![Some("blind"), Some("informed")].contains(&v.as_str()))
                || escalation
                    .get("on_low_confidence")
                    .is_some_and(|v| ![Some("error"), Some("random")].contains(&v.as_str()))
            {
                return bad();
            }
        }
        Ok(project)
    }
    pub fn route(&self, name: &str, kind: Option<&str>) -> Res<(J, J)> {
        let m = self.data["routes"]
            .get(name)
            .and_then(J::as_str)
            .and_then(|m| self.data["models"].get(m));
        let Some(m) = m else {
            return fail("ConfigurationError", "unknown configured model route");
        };
        if kind.is_some_and(|k| m["kind"] != k) {
            return fail("ConfigurationError", "model route has wrong kind");
        }
        Ok((
            self.data["providers"][m["provider"].as_str().unwrap()].clone(),
            m.clone(),
        ))
    }
    pub fn timeout(&self, model: &J, override_value: Option<Duration>) -> Duration {
        override_value.unwrap_or_else(|| {
            Duration::from_secs_f64(
                model["timeout"]
                    .as_f64()
                    .or_else(|| self.data["runtime"]["timeout"].as_f64())
                    .unwrap_or(30.),
            )
        })
    }
}
