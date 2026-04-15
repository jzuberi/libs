from workflows.engine.call_workflow import call_workflow

# This wraps the simple_project workflow as a step
call_simple_step = call_workflow("simple_project")
