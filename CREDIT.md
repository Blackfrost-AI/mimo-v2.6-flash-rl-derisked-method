# Credit

The workflow and recipes in this package are **derived from Drowzeys' work**.

- The MoE expert-redirection (expert-picking) method for removing refusal
  behavior is Drowzeys' recipe.
- The refusal-transition-boundary capture-and-freeze procedure (capturing the
  model's own reasoning-transition boundaries, then freezing them into a
  runtime redirect map) is Drowzeys' recipe.
- The λ-strength intervention format (`pass*-drowzeys-experts-l*-*lambda*.json`)
  follows his recipe structure; our final map
  (`pass12-drowzeys-experts-l28-33-iteration2-lambda3.5.json`) is named for it.

This project adapted those recipes to XiaomiMiMo/MiMo-V2.6-Flash-RL and
combined them with a chat-template bake of the lab-authorization system prompt
(our addition). The serving-time bake + render-equivalence test + kill-switch
evaluation harness are ours; the core derisking workflow is Drowzeys'.

All credit for the underlying method goes to Drowzeys.
