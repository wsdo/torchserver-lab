

# Existing content from opan_ai.py
{opan_ai_content}

# Additional imports and initialization from chatglm_handler.py
{initialization_code}

# FastAPI endpoint for generating answers
@app.post("/generate")
def generate_answer(context: str, question: str):
    # Create an instance of the handler
    handler = ChatGLMHandler()
    
    # Dummy context for initialization (actual model directory would be used in real deployment)
    dummy_ctx = type('DummyContext', (object,), {'manifest': {}, 'system_properties': {'model_dir': 'path_to_model'}})()
    handler.initialize(dummy_ctx)
    
    # Preprocess
    data = [{"data": {"context": context, "question": question}}]
    model_input = handler.preprocess(data)
    
    # Inference
    model_output = handler.inference(model_input)
    
    # Postprocess
    response = handler.postprocess(model_output)
    
    return response[0]

