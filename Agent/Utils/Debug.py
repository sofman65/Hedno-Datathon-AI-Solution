import inspect

def print_caller_info(variable):
    
    # Get the current frame
    current_frame = inspect.currentframe()
    
    # Get the frame of the caller
    caller_frame = inspect.getouterframes(current_frame, 2)[1][0]
    
    # Extract caller information
    caller_info = inspect.getframeinfo(caller_frame)
    
    # Get the name of the class, if available
    class_name = ''
    if 'self' in caller_frame.f_locals:
        class_name = caller_frame.f_locals['self'].__class__.__name__
    
    # Print the requested information
    print(f"Variable: {variable}")
    print(f"File: {caller_info.filename}")
    print(f"Line: {caller_info.lineno}")
    print(f"Class: {class_name}")
    print(f"Method: {caller_info.function}")
