# Field notes: a court for cats

An early end-to-end experiment built a fictional cat tribunal in Purrhaps.
Hosted generative models wrote prosecution and defense speeches, a local
decision model selected verdicts from fixed options, and a final generative
call wrote a tabloid headline. A completed four-case session used 21 model calls.

The experiment helped expose several distinctions the language needs to make.

- **Sampling is not acceptance.** The application used `.decision` to freeze
  one low-confidence answer. That motivated `sample(x)`, which preserves a
  snapshot without requiring it to pass the confidence gate.
- **Confidence is not option probability.** The initial application tested the
  largest option score against 0.8, while the runtime tested the provider's
  separate confidence value. Those policies can produce different outcomes.
  A displayed “80% sure” was not proof the runtime would accept the verdict.
- **A gate on `if` is not a gate on every effect.** Map lookup and `print` could
  produce verdict text where a branch would refuse to run. The runtime tracks
  value provenance; it does not provide a universal effect system.
- **Task wording matters.** Some models echoed a String input under the old
  instruction to interpret it as a String. That behavior did not isolate model
  weakness: the task and adapter instructions were ambiguous. Structured
  conversation inputs helped, and the adapter instructions were clarified.
- **Validation can interrupt the story.** A nonexistent source citation raised
  `ConfabulationError`. Later prompt changes meant that counting failures across
  all development runs did not measure the final version's reliability.
- **Uncertainty survives generation.** The headline inherited uncertainty from
  its input verdicts. A fluent sentence did not manufacture confidence in its
  underlying claims.

These observations led to [decision-first escalation](escalation.md): a decision
model gets the first attempt, a generative model may follow, and an explicit
random terminal policy can authorize progress if neither reaches the threshold.
The result records how it was selected. Random selection reports zero evidential
confidence and keeps its selection probability separate.

The accompanying tests use scripted adapters, local HTTP fixtures and fake
harnesses. Those tests establish runtime behavior and Python/native parity, not
the correctness of fictional verdicts or the reliability of any live provider.
