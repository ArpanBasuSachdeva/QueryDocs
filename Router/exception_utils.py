import traceback
from .database import ExceptionLog

# Utility function to log exceptions to the database
def log_exception(error, function_name=None, context=None, session=None, user_id=None):
    """
    Logs an exception to the ExceptionLog table.
    Args:
        error: Exception object
        function_name: Name of the function where the error occurred
        context: Additional context about the error
        session: SQLAlchemy session object
        user_id: (Optional) ID of the user related to the error
    """
    if session:
        error_message = str(error)
        stack_trace = traceback.format_exc()
        exception_log = ExceptionLog(
            error_message=error_message,
            stack_trace=stack_trace,
            user_id=user_id
        )
        session.add(exception_log)
        session.commit()
    else:
        # If no session is provided, just print the error
        print(f"Error in {function_name}: {str(error)}")
        if context:
            print(f"Context: {context}")
        print(traceback.format_exc()) 