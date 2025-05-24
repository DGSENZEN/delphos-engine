import { type NextRequest, NextResponse } from "next/server"
import DOMPurify from "dompurify"
import { JSDOM } from "jsdom"
import { API_CONFIG } from "@/lib/config"

// This is needed for DOMPurify to work in a Node.js environment
const window = new JSDOM("").window
const purify = DOMPurify(window)

export async function POST(request: NextRequest) {
  try {
    // Parse the request body
    const formData = await request.formData()
    const topic = formData.get("topic") as string
    const threadId = formData.get("thread_id") as string | null
    const files = formData.getAll("media_files") as File[]

    console.log("Received deep research request with topic:", topic)
    console.log("Thread ID:", threadId || "none")
    console.log("Files:", files.length > 0 ? files.map((f) => f.name).join(", ") : "none")

    if (!topic) {
      return NextResponse.json({ error: "Topic parameter is required" }, { status: 400 })
    }

    // Get the start time to calculate response time
    const startTime = Date.now()

    try {
      // Create a new FormData object to send to the API
      const apiFormData = new FormData()

      // Your FastAPI expects 'topic'
      apiFormData.append("topic", topic)

      if (threadId) {
        apiFormData.append("thread_id", threadId)
      }

      // Add any files to the form data with the EXACT name expected by FastAPI
      files.forEach((file) => {
        apiFormData.append("media_files", file)
      })

      // Construct the API endpoint URL using the config
      const fullApiUrl = `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.deepResearch}`

      console.log(`Making request to: ${fullApiUrl}`)
      console.log(`Request topic: ${topic}`)
      console.log(`Request files: ${files.length}`)

      // Log the form data entries for debugging
      console.log(
        "Form data entries:",
        [...apiFormData.entries()].map(([key, value]) =>
          typeof value === "string" ? `${key}: ${value}` : `${key}: [File: ${(value as File).name}]`,
        ),
      )

      // Make the request to the FastAPI backend
      const response = await fetch(fullApiUrl, {
        method: "POST",
        body: apiFormData,
        // Add headers that might help with CORS or content negotiation
        headers: {
          Accept: "application/json",
        },
      })

      console.log(`API response status: ${response.status}`)

      // Handle rate limiting (429 status code)
      if (response.status === 429) {
        return NextResponse.json(
          {
            content: `<p class="text-slate-200 font-light mb-4">Rate limit exceeded. Please try again in a minute.</p>`,
            responseTime: ((Date.now() - startTime) / 1000).toFixed(1),
            additionalInfo: ["Rate limit: 3 requests per minute for deep research"],
            error: true,
          },
          { status: 429 },
        )
      }

      if (!response.ok) {
        const errorText = await response.text()
        console.error(`API error response (${response.status}):`, errorText)

        // Try to parse the error response as JSON if possible
        let errorDetail = errorText
        try {
          const errorJson = JSON.parse(errorText)
          errorDetail = errorJson.detail || errorJson.message || errorText
        } catch (e) {
          // Not JSON, use the text as is
        }

        throw new Error(`API responded with status: ${response.status}, message: ${errorDetail}`)
      }

      // Get the response text first to check if it's valid JSON
      const responseText = await response.text()
      console.log("API response text (first 100 chars):", responseText.substring(0, 100))

      let apiData
      try {
        apiData = JSON.parse(responseText)
      } catch (jsonError) {
        console.error("Error parsing JSON response:", jsonError, "Response text:", responseText.substring(0, 200))
        return NextResponse.json(
          {
            content: `<p class="text-slate-200 font-light mb-4">Error parsing API response. The API returned invalid JSON.</p>
                     <p class="text-slate-200 font-light mb-4">Response: ${responseText.substring(0, 200)}...</p>`,
            responseTime: ((Date.now() - startTime) / 1000).toFixed(1),
            additionalInfo: ["API returned invalid JSON"],
            error: true,
          },
          { status: 500 },
        )
      }

      // The FastAPI returns { response: string, thread_id: string } format
      const responseContent = apiData.response || ""
      const responseThreadId = apiData.thread_id || ""

      // Calculate response time in seconds
      const responseTime = ((Date.now() - startTime) / 1000).toFixed(1)

      // Format the response for our frontend
      const formattedResponse = {
        content: formatContent(responseContent),
        responseTime,
        threadId: responseThreadId,
        additionalInfo: ["Deep research mode"],
      }

      return NextResponse.json(formattedResponse)
    } catch (apiError: any) {
      console.error("API request error:", apiError.message || apiError)

      // Use mock response if we're in development and the API is not available
      if (process.env.USE_MOCK_RESPONSES === "true") {
        console.log("Using mock response in development mode")
        return NextResponse.json({
          content: `<p class="text-slate-200 font-light mb-4">This is a mock deep research response for development. Your topic was: "${topic}"</p>
                    <p class="text-slate-200 font-light mb-4">In a production environment, this would connect to your API.</p>`,
          responseTime: 2.5,
          threadId: "mock_thread_" + Date.now().toString(36),
          additionalInfo: ["Development mode active", "Using mock response", `Topic was: "${topic}"`],
        })
      }

      return NextResponse.json(
        {
          content: `<p class="text-slate-200 font-light mb-4">Error connecting to the API: ${apiError.message || "Unknown error"}</p>
                    <p class="text-slate-200 font-light mb-4">Please check that your API server is running and accessible.</p>`,
          responseTime: ((Date.now() - startTime) / 1000).toFixed(1),
          additionalInfo: [
            "Error details: " + (apiError.message || "Unknown error"),
            `API endpoint: ${API_CONFIG.baseUrl}${API_CONFIG.endpoints.deepResearch}`,
            `Topic was: "${topic}"`,
          ],
          error: true,
        },
        { status: 500 },
      )
    }
  } catch (error: any) {
    console.error("Deep research API error:", error.message || error)
    return NextResponse.json(
      {
        content: `<p class="text-destructive">Failed to process deep research request: ${error.message || "Unknown error"}</p>`,
        error: true,
      },
      { status: 500 },
    )
  }
}

// Helper function to format content with appropriate HTML/CSS for this project
function formatContent(content: string): string {
  // Sanitize the content
  const sanitizedContent = purify.sanitize(content)

  return sanitizedContent
    .replace(/<p>/g, '<p class="text-slate-200 font-light mb-4">')
    .replace(/<h(\d)>/g, '<h$1 class="text-slate-100 font-medium mb-3">')
    .replace(/<ul>/g, '<ul class="list-disc pl-5 mb-4 text-slate-300">')
    .replace(/<li>/g, '<li class="mb-2">')
    .replace(/<a /g, '<a class="text-orange-500 hover:text-orange-400 underline" ')
    .replace(/<code>/g, '<code class="bg-slate-800 px-1 py-0.5 rounded text-teal-500">')
}
