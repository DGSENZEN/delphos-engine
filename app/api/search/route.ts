import { type NextRequest, NextResponse } from "next/server"
import DOMPurify from "dompurify"
import { JSDOM } from "jsdom"
import { API_CONFIG } from "@/lib/config" // Assuming this contains baseUrl and endpoint paths

// This is needed for DOMPurify to work in a Node.js environment
const window = new JSDOM("").window
const purify = DOMPurify(window)

export async function POST(request: NextRequest) {
  const startTime = Date.now() // Start timer early for accurate error timing
  let query: string | null = null // Initialize query variable

  try {
    // Parse the request body
    const formData = await request.formData()
    query = formData.get("query") as string // Get query early for error logging
    const threadId = formData.get("thread_id") as string | null
    const files = formData.getAll("media_files") as File[]
    // Get the deep research flag - formData values are strings or Files
    const isDeepResearch = formData.get("isDeepResearch") === "true"

    console.log(`Received request: query="${query}", isDeepResearch=${isDeepResearch}`)
    console.log("Thread ID:", threadId || "none")
    console.log("Files:", files.length > 0 ? files.map((f) => f.name).join(", ") : "none")

    if (!query) {
      return NextResponse.json({ error: "Query parameter is required" }, { status: 400 })
    }

    // --- Determine target endpoint and primary text field name ---
    let targetEndpointPath: string
    let primaryTextFieldName: string

    if (isDeepResearch) {
      targetEndpointPath = API_CONFIG.endpoints.deep_research // e.g., '/deep_research'
      primaryTextFieldName = "topic" // FastAPI expects 'topic' for this endpoint
      console.log("Mode: Deep Research - Sending field 'topic'")
    } else {
      targetEndpointPath = API_CONFIG.endpoints.query // e.g., '/query'
      // --- FIX: Send 'query' to match FastAPI alias ---
      // Even though the FastAPI param is text_input, the alias="query"
      // and the 422 error indicate FastAPI validation expects 'query'.
      primaryTextFieldName = "query"
      console.log("Mode: Standard Query - Sending field 'query' (matching alias)")
      // ---------------------------------------------
    }
    // -------------------------------------------------------------

    // Ensure the endpoint path exists in config
    if (!targetEndpointPath) {
        throw new Error(`API endpoint path for ${isDeepResearch ? 'deep_research' : 'query'} is not defined in API_CONFIG.`);
    }

    // Create a new FormData object to send to the API
    const apiFormData = new FormData()

    // Append the primary text field with the correct name
    apiFormData.append(primaryTextFieldName, query)

    // Append thread_id if present
    if (threadId) {
      apiFormData.append("thread_id", threadId)
    }

    // Append files with the name FastAPI expects
    files.forEach((file) => {
      apiFormData.append("media_files", file)
    })

    // Construct the full API endpoint URL
    const fullApiUrl = `${API_CONFIG.baseUrl}${targetEndpointPath}`

    console.log(`Making POST request to: ${fullApiUrl}`)
    // Log the form data entries for debugging
    console.log(
      "Form data entries being sent:",
      [...apiFormData.entries()].map(([key, value]) =>
        typeof value === "string" ? `${key}: ${value}` : `${key}: [File: ${(value as File).name}]`,
      ),
    )

    // Make the request to the FastAPI backend
    const response = await fetch(fullApiUrl, {
      method: "POST",
      body: apiFormData,
      headers: {
        Accept: "application/json",
      },
      cache: 'no-store', // Prevent caching of API responses
    })

    console.log(`API response status: ${response.status}`)

    // --- Handle Response (Rate Limit, Errors, Success) ---
    if (response.status === 429) {
      // Handle rate limiting
      const rateLimitMessage = isDeepResearch
        ? "Deep research rate limit exceeded (3/min). Please wait."
        : "Rate limit exceeded (10/min). Please try again shortly."
      return NextResponse.json(
        {
          content: `<p class="text-slate-200 font-light mb-4">${rateLimitMessage}</p>`,
          responseTime: ((Date.now() - startTime) / 1000).toFixed(1),
          additionalInfo: ["Rate limit exceeded"],
          error: true,
        },
        { status: 429 },
      )
    }

    if (!response.ok) {
      // Handle other errors
      const errorText = await response.text()
      console.error(`API error response (${response.status}):`, errorText)
      let errorDetail = errorText
      try {
        const errorJson = JSON.parse(errorText)
        errorDetail = errorJson.detail || errorJson.message || errorText
      } catch (e) { /* Not JSON */ }
      // If it was a 422, make the error message clearer
      if (response.status === 422) {
         errorDetail = `Invalid data sent to API: ${errorDetail}`;
      }
      throw new Error(`API responded with status: ${response.status}, message: ${errorDetail}`)
    }

    // --- Process Successful Response ---
    const responseText = await response.text()
    console.log("API response text (first 100 chars):", responseText.substring(0, 100))

    let apiData
    try {
      apiData = JSON.parse(responseText)
    } catch (jsonError) {
      // Handle cases where API gives 200 OK but invalid JSON
      console.error("Error parsing JSON response:", jsonError, "Response text:", responseText.substring(0, 200))
      return NextResponse.json(
        {
          content: `<p class="text-slate-200 font-light mb-4">Error parsing API response. The API returned invalid JSON despite a success status.</p>
                   <p class="text-slate-200 font-light mb-4">Response: ${responseText.substring(0, 200)}...</p>`,
          responseTime: ((Date.now() - startTime) / 1000).toFixed(1),
          additionalInfo: ["API returned invalid JSON"],
          error: true,
        },
        { status: 500 }, // Treat invalid JSON on success as a server error
      )
    }

    const responseContent = apiData.response || ""
    const responseThreadId = apiData.thread_id || ""
    const responseTime = ((Date.now() - startTime) / 1000).toFixed(1)

    const formattedResponse = {
      content: formatContent(responseContent), // Sanitize and format
      responseTime,
      threadId: responseThreadId,
      additionalInfo: [], // Add any relevant info if needed
    }

    return NextResponse.json(formattedResponse)

  } catch (error: any) {
    // --- Catch errors from fetch itself or other processing ---
    console.error("Error in /api/search route:", error.message || error)

    // Mock response logic (keep if needed for development)
    if (process.env.USE_MOCK_RESPONSES === "true") {
      console.log("Using mock response due to error or mock mode")
      // Ensure query is defined even if error happened early
      const currentQuery = query || "[Query not available]";
      return NextResponse.json({
        content: `<p class="text-slate-200 font-light mb-4">This is a mock response. Your query was: "${currentQuery}"</p>`,
        responseTime: 0.1,
        threadId: "mock_thread_" + Date.now().toString(36),
        additionalInfo: ["Development mode active", "Using mock response", `Query was: "${currentQuery}"`],
      })
    }

    // Return generic error response
    const errorMessage = error instanceof Error ? error.message : "Unknown error processing search";
    return NextResponse.json(
      {
        content: `<p class="text-slate-200 font-light mb-4">Error processing search: ${errorMessage}</p>
        <p class="text-slate-200 font-light mb-4">Please check the API server logs for more details.</p>`,
        responseTime: ((Date.now() - startTime) / 1000).toFixed(1),
        additionalInfo: [
          "Error details: " + errorMessage,
          `Query was: "${query || "[Query not available]"}"`,
        ],
        error: true,
      },
      { status: 500 },
    )
  }
}

// Helper function to format content (keep as is)
function formatContent(content: string): string {
  // Basic check if content might be HTML already
  if (content && content.trim().startsWith('<') && content.trim().endsWith('>')) {
    // If it looks like HTML, sanitize it directly
    return purify.sanitize(content);
  }
  // Otherwise, assume it's plain text or Markdown that needs basic formatting
  // (Ideally, use a proper Markdown renderer if your API returns Markdown)
  const sanitizedContent = purify.sanitize(content || ""); // Sanitize plain text too
  return sanitizedContent
    .replace(/<p>/g, '<p class="text-slate-200 font-light mb-4">') // Apply classes if tags exist after sanitize
    .replace(/<h(\d)>/g, '<h$1 class="text-slate-100 font-medium mb-3">')
    .replace(/<ul>/g, '<ul class="list-disc pl-5 mb-4 text-slate-300">')
    .replace(/<li>/g, '<li class="mb-2">')
    .replace(/<a /g, '<a class="text-orange-500 hover:text-orange-400 underline" ')
    .replace(/<code>/g, '<code class="bg-slate-800 px-1 py-0.5 rounded text-teal-500">')
    // Add a basic paragraph wrap if no tags were present
    .replace(/^([^<].*[^>])$/gm, '<p class="text-slate-200 font-light mb-4">$1</p>');
}
