# Tool Calling

Gemini receives only declarations created by `ToolRegistry`. The application executes calls backend-side. The registry rejects non-allowlisted names, unknown arguments and missing required arguments. Agent step and call limits are configured centrally.

This separation matters because model intent is not authorization. A model may propose a function call; only deterministic backend validation can authorize it.
