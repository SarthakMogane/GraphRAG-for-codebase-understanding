import traceback

try:
    stream_result = self.llm.astream(messages=[...])
    print("DEBUG: stream_result created successfully")
    
    async for chunk in stream_result:
        print(f"DEBUG: chunk={repr(chunk)}")
        print(f"DEBUG: type(chunk)={type(chunk)}")
        print(f"DEBUG: hasattr(chunk, 'get')={hasattr(chunk, 'get')}")
        break  # Just first chunk
        
except Exception as e:
    logger.error(f"FULL TRACEBACK:\n{traceback.format_exc()}")
