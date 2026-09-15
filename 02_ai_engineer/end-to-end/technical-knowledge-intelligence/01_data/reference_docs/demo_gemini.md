# Gemini Structured Outputs and Tool Calling

Structured output constrains the model response to a schema so application code can validate and render predictable fields. Function calling is different: the model selects a declared function and arguments, but the application executes the function. A safe tool layer should allowlist names, validate arguments and bound the number of calls.
