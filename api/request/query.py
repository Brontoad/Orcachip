from api.request import request
class QueryRequest(request.Request): 
    """
        Requests containing the search query.
        Properties:
            query: str
    """
    query: str