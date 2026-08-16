const search = async () => {
    let query = document.getElementById("query").value;

    try {
        const response = await fetch("/search", {
            method: "post", 
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({"query": query}),
        })   

        if (!response.ok) { throw new Error(`Response status: ${response.status}`); }

        const searchResults = await response.json()
        
        displaySearchResults(searchResults)
    } catch (error) { console.log(error.message) }
}


const displaySearchResults = (searchResults) => {
    let searchResultsDiv = document.getElementById("search-results");
    console.log(searchResults)
}