import { NextResponse } from "next/server"
import { API_CONFIG } from "@/lib/config"

export async function GET(request) {
  try {
    // Check if we should use mock responses
    const useMockResponses = process.env.USE_MOCK_RESPONSES === "true" || false

    if (useMockResponses) {
      return NextResponse.json({ status: "mock", message: "Using mock responses" })
    }

    // Construct the health endpoint URL using the config
    const healthEndpoint = `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.health}`

    console.log(`Checking API health at: ${healthEndpoint}`)

    try {
      // Make a single request with a longer timeout
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 10000) // 10-second timeout

      const response = await fetch(healthEndpoint, {
        method: "GET",
        signal: controller.signal,
        cache: "no-store",
        headers: {
          Accept: "application/json",
        },
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        throw new Error(`Health check failed with status: ${response.status}`)
      }

      // Get the response text first to check if it's valid JSON
      const responseText = await response.text()
      console.log("API response text:", responseText)

      let data
      try {
        data = JSON.parse(responseText)
      } catch (jsonError) {
        console.error("Error parsing JSON response:", jsonError, "Response text:", responseText)
        return NextResponse.json({ status: "error", message: "API responded but not with valid JSON" }, { status: 500 })
      }

      return NextResponse.json({ status: data.status || "ok" })
    } catch (fetchError) {
      console.error("Health check fetch error:", fetchError.message || fetchError)
      return NextResponse.json(
        { status: "error", message: `API connection error: ${fetchError.message || "Unknown error"}` },
        { status: 503 },
      )
    }
  } catch (error) {
    console.error("Health check error:", error.message || error)
    return NextResponse.json(
      { status: "error", message: error.message || "Failed to check API health" },
      { status: 503 },
    )
  }
}
