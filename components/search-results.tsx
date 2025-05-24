"use client"

import { useState, useEffect, useRef } from "react"
import MarkdownRenderer from "./markdown-renderer" // Assuming you have this component

interface SearchResultsProps {
  query: string
  files: File[] // Changed from optional to required based on usage
  useMockResponse?: boolean
  isDeepResearch?: boolean
  threadId?: string // Changed from optional to required based on usage
}

interface ResponseData {
  content: string
  responseTime?: number | string // Allow string for formatted time
  additionalInfo?: string[]
  threadId?: string
  error?: boolean
}

// Mock responses for offline mode
const MOCK_RESPONSES: Record<string, string> = {
  default: `# Data Analysis Report (Mock)

Based on your query, the following information has been retrieved:

Large Language Models (LLMs) have been applied to various domains including healthcare, legal document analysis, code generation, and scientific research. Recent papers have explored their use in drug discovery, protein folding prediction, and automated reasoning.

## Research Findings

- Applications of LLMs in scientific discovery processes
- Enhancing LLM reasoning through chain-of-thought techniques
- Multimodal models that combine text, image, and audio understanding
- Specialized domain adaptation for medical and legal applications

## Implementation Example

\`\`\`python
def analyze_text(text, model="gpt-4"):
    """
    Analyze text using a large language model
    """
    response = llm_api.generate(
        prompt=text,
        model=model,
        max_tokens=1000
    )
    return response.text
\`\`\`

[NOTICE: This is a simulated response generated while in offline mode]`,
  deep_research: `# Comprehensive Research Report (Mock)

## Topic Overview
This is an in-depth analysis of the requested topic, synthesizing information from multiple sources.

## Key Findings
- Finding 1: Detailed explanation with supporting evidence
- Finding 2: Analysis of current research and historical context
- Finding 3: Evaluation of competing theories and methodologies

## Expert Perspectives
Various domain experts have contributed different viewpoints on this topic:

1. **Perspective A**: Main arguments and supporting evidence
2. **Perspective B**: Alternative interpretation with case studies
3. **Perspective C**: Critical analysis and limitations

## Practical Applications
- Application 1: How these findings can be implemented
- Application 2: Potential benefits and challenges
- Application 3: Future directions and opportunities

[NOTICE: This is a simulated deep research response generated while in offline mode]`,
}

export default function SearchResults({
  query,
  files = [], // Default to empty array if not provided
  useMockResponse = false,
  isDeepResearch = false, // Default to false
  threadId, // Keep optional, handle undefined below
}: SearchResultsProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [responseData, setResponseData] = useState<ResponseData | null>(null)
  const [typingEffect, setTypingEffect] = useState(false)
  const [displayedContent, setDisplayedContent] = useState("")
  const [fullContent, setFullContent] = useState("")
  const [isMarkdown, setIsMarkdown] = useState(false)
  const [rateLimited, setRateLimited] = useState(false)
  // Use state for threadId to manage it across requests within this component instance
  const [currentThreadId, setCurrentThreadId] = useState<string | undefined>(threadId)
  const [requestDetails, setRequestDetails] = useState<string | null>(null)
  const [responseDetails, setResponseDetails] = useState<string | null>(null)

  const abortControllerRef = useRef<AbortController | null>(null)
  const isFetchingRef = useRef(false)
  // Use a ref to track the query that triggered the current/last fetch
  const currentFetchQueryRef = useRef<string>("")

  // Cleanup function
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        console.log("Aborting fetch on component unmount or query change")
        abortControllerRef.current.abort()
      }
    }
  }, [])

  // Fetch results when the query, files, or mode changes
  useEffect(() => {
    // Skip if query is empty or if already fetching for this exact query/mode
    if (!query || (query === currentFetchQueryRef.current && isFetchingRef.current)) {
      return
    }

    const fetchResults = async () => {
      // Set fetching state for the current query
      isFetchingRef.current = true
      currentFetchQueryRef.current = query // Track the query being fetched

      // Abort any previous request before starting a new one
      if (abortControllerRef.current) {
        console.log("Aborting previous fetch request")
        abortControllerRef.current.abort("New request started") // Provide reason
      }

      // Create a new abort controller for this request
      abortControllerRef.current = new AbortController()

      // Reset state for the new request
      setLoading(true)
      setError(null)
      setTypingEffect(false)
      setDisplayedContent("")
      setFullContent("") // Clear previous full content
      setIsMarkdown(false)
      setRateLimited(false)
      setRequestDetails(null)
      setResponseDetails(null)
      setResponseData(null) // Clear previous response data

      try {
        console.log("Fetching results for query:", query, "Mock mode:", useMockResponse)
        console.log("Files:", files.length > 0 ? files.map((f) => f.name).join(", ") : "None")
        console.log("Mode:", isDeepResearch ? "Deep Research" : "Standard Query")
        console.log("Thread ID:", currentThreadId || "New thread")

        const startTime = Date.now()

        // --- Mock Response Handling ---
        if (useMockResponse) {
          await new Promise((resolve) => setTimeout(resolve, isDeepResearch ? 3000 : 1500)) // Simulate delay
          if (abortControllerRef.current?.signal.aborted) return // Check if aborted during delay

          const mockContent = isDeepResearch ? MOCK_RESPONSES.deep_research : MOCK_RESPONSES.default
          const mockThreadId = currentThreadId || `mock_thread_${Date.now().toString(36)}`

          setResponseData({
            content: mockContent,
            responseTime: isDeepResearch ? 3.5 : 1.2,
            additionalInfo: ["Using offline mode", isDeepResearch ? "Deep research mode" : "Standard query mode"],
            threadId: mockThreadId,
          })
          setCurrentThreadId(mockThreadId) // Persist mock thread ID
          setFullContent(mockContent)
          setIsMarkdown(true) // Assume mock responses are markdown
          setTypingEffect(true)
          setLoading(false)
          isFetchingRef.current = false // Reset fetching state
          return
        }

        // --- Real API Request ---
        const formData = new FormData()

        // Append data expected by the Next.js API route (/api/search or /api/deep-research)
        formData.append("query", query) // Next.js route expects 'query' for both modes
        formData.append("isDeepResearch", String(isDeepResearch)) // Send boolean as string

        if (currentThreadId) {
          formData.append("thread_id", currentThreadId)
        }

        files.forEach((file) => {
          // Ensure the key matches what the Next.js route expects
          formData.append("media_files", file)
        })

        // Log form data being sent to the Next.js route
        const formDataEntries = [...formData.entries()].map(([key, value]) =>
          typeof value === "string" ? `${key}: ${value}` : `${key}: [File: ${(value as File).name}]`,
        )
        console.log("Form data entries sent to Next.js route:", formDataEntries)
        setRequestDetails(`Sending: ${formDataEntries.join(", ")}`)

        // Determine the correct Next.js API endpoint
        // IMPORTANT: Ensure you have corresponding files like /app/api/search/route.js and /app/api/deep-research/route.js
        // For simplicity now, let's assume you have ONE route /api/search that handles both based on the isDeepResearch flag
        const endpoint = "/api/search" // Using a single endpoint is often cleaner
        // const endpoint = isDeepResearch ? "/api/deep-research" : "/api/search"; // Use this if you have separate routes

        console.log(`Making request to Next.js route: ${endpoint}`)

        const response = await fetch(endpoint, {
          method: "POST",
          body: formData,
          signal: abortControllerRef.current.signal, // Pass the signal
        })

        // Check if the request was aborted *after* the fetch started but before completion
        if (abortControllerRef.current?.signal.aborted) {
            console.log("Fetch aborted after starting.");
            // Don't proceed with processing the response
            // Reset fetching state appropriately if needed, though the finally block handles it
            return;
        }


        console.log("Response status from Next.js route:", response.status)
        setResponseDetails(`Response status: ${response.status}`)

        if (response.status === 429) {
          setRateLimited(true)
          throw new Error("Rate limit exceeded. Please try again in a minute.")
        }

        if (!response.ok) {
          const errorText = await response.text()
          console.error("API error response from Next.js route:", errorText)
          let errorDetail = errorText
          try {
            const errorJson = JSON.parse(errorText)
            // Use the message from the Next.js route's error response
            errorDetail = errorJson.message || errorJson.detail || errorText
          } catch (e) { /* Not JSON */ }
          setResponseDetails(`Error response: ${errorDetail}`)
          throw new Error(`API route responded with status: ${response.status}, message: ${errorDetail}`)
        }

        const responseTextFull = await response.text()
        console.log("Response text from Next.js route (first 100 chars):", responseTextFull.substring(0, 100))
        setResponseDetails(`Response received (${responseTextFull.length} bytes)`)

        let data: ResponseData
        try {
          data = JSON.parse(responseTextFull)
          console.log("Parsed response data from Next.js route:", data)
        } catch (e) {
          console.error("Failed to parse response JSON from Next.js route:", e)
          setResponseDetails(`Failed to parse JSON: ${responseTextFull.substring(0, 100)}...`)
          throw new Error(`Invalid JSON received from API route: ${responseTextFull.substring(0, 100)}...`)
        }

        // Update thread ID if received
        if (data.threadId) {
          setCurrentThreadId(data.threadId)
        }

        // Check if the response content looks like markdown
        const responseContent = data.content || ""
        const isMarkdownResponse =
          responseContent.includes("# ") ||
          responseContent.includes("## ") ||
          responseContent.includes("```") ||
          (responseContent.includes("*") && responseContent.includes("*")) ||
          responseContent.includes("[Source URL:") // Check for your citation format

        setIsMarkdown(isMarkdownResponse)
        setResponseData(data) // Store the entire data object
        setFullContent(responseContent)
        setTypingEffect(true) // Start typing effect

      } catch (err: any) {
         // Only set error state if the error wasn't an abort
        if (err.name !== "AbortError") {
            console.error("Error fetching search results:", err)
            setError(`Failed to fetch results: ${err.message || "Unknown error"}. Please try again.`)
            // Clear previous successful response data on error
            setResponseData(null)
            setFullContent("")
            setDisplayedContent("")
        } else {
            console.log("Fetch operation was aborted successfully.");
            // Optionally reset loading state here if needed, though finally block handles it
        }
      } finally {
        setLoading(false)
        isFetchingRef.current = false // Reset fetching state regardless of outcome
        // Clean up the abort controller ref after the fetch attempt is fully finished
        abortControllerRef.current = null;
      }
    }

    fetchResults()

    // Cleanup function for this specific effect run
    return () => {
        if (abortControllerRef.current) {
            console.log("Aborting fetch on effect cleanup (query/files/mode change)");
            abortControllerRef.current.abort("Component re-rendered or dependency changed");
        }
    }
  // Rerun effect if query, files array reference, mock status, or research mode changes
  // Note: Comparing files array by reference might not be ideal if the parent component recreates the array.
  // Consider stringifying or using a stable key if files cause unnecessary refetches.
  }, [query, files, useMockResponse, isDeepResearch]) // Removed threadId from deps, managed by currentThreadId state

  // Typing effect logic (remains the same)
  useEffect(() => {
    if (!typingEffect || !fullContent) {
        setDisplayedContent(fullContent); // Ensure full content is shown if typing effect is off or content is empty
        return;
    }

    if (isMarkdown) {
      setDisplayedContent(fullContent) // Show markdown immediately
      return
    }

    let typingInterval: NodeJS.Timeout | null = null;
    let currentIndex = 0
    const baseSpeed = 5
    const variableSpeed = () => Math.max(2, baseSpeed - (currentIndex / fullContent.length) * 3)

    const typeCharacter = () => {
        if (currentIndex < fullContent.length) {
            setDisplayedContent(fullContent.substring(0, currentIndex + 1));
            currentIndex++;
            typingInterval = setTimeout(typeCharacter, variableSpeed());
        } else {
            // Typing finished
        }
    };

    // Start typing after a short delay
    const startTimeout = setTimeout(() => {
        typeCharacter();
    }, 100);


    return () => {
      if (typingInterval) clearTimeout(typingInterval)
      clearTimeout(startTimeout);
    }
  }, [typingEffect, fullContent, isMarkdown])

  // Formatting function (remains the same)
  function formatContent(content: string): string {
    // ... (keep your existing formatContent logic) ...
     if (!content) return ""; // Handle empty content
    // Basic check if it looks like HTML already
    if (content.trim().startsWith("<") && content.trim().endsWith(">")) {
        // Assume it's already formatted or sanitized HTML from the API route
        return content;
    }
    // Fallback basic formatting if needed (though MarkdownRenderer should handle most)
    return content
        .split('\n\n')
        .map(paragraph => `<p class="text-foreground font-light mb-4">${paragraph.replace(/\n/g, '<br/>')}</p>`)
        .join('');
  }

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Query Card (remains the same) */}
      <div className="nier-container mb-4">
        <div className="nier-header">
          <div className="nier-title">{isDeepResearch ? "DEEP RESEARCH TOPIC" : "QUERY"}</div>
          <div className="text-xs uppercase tracking-wider">
            ID:{" "}
            {currentThreadId
              ? currentThreadId.slice(-6).toUpperCase()
              : "N/A"} {/* Show N/A if no thread ID yet */}
          </div>
        </div>
        <div className="nier-content">
          <p className="text-foreground font-mono uppercase">{query}</p>
          {files.length > 0 && (
            <div className="mt-2 pt-2 border-t border-border">
              <p className="text-xs uppercase tracking-wider text-muted-foreground mb-2">Attachments:</p>
              <div className="flex flex-wrap gap-2">
                {files.map((file, index) => (
                  <div key={index} className="text-xs bg-muted px-2 py-1 rounded flex items-center gap-1">
                    <span>{file.name}</span>
                    <span className="text-muted-foreground">({(file.size / 1024).toFixed(1)} KB)</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {/* Debug details */}
          {requestDetails && <div className="mt-2 pt-2 border-t border-border"><p className="text-xs text-muted-foreground">{requestDetails}</p></div>}
          {responseDetails && <div className="mt-2 pt-2 border-t border-border"><p className="text-xs text-muted-foreground">{responseDetails}</p></div>}
        </div>
      </div>

      {/* Response Card (remains largely the same, uses MarkdownRenderer) */}
      <div className="nier-container">
        <div className="nier-header">
          <div className="flex items-center gap-3">
            <div className="nier-title">RESPONSE</div>
            {useMockResponse && <span className="text-xs px-2 py-0.5 bg-muted text-secondary uppercase tracking-wider">Offline Mode</span>}
            {isDeepResearch && <span className="text-xs px-2 py-0.5 bg-muted text-secondary uppercase tracking-wider">Deep Research</span>}
          </div>
          <div className="text-xs uppercase tracking-wider">
            {loading ? "Processing..." : responseData?.responseTime ? `Response time: ${responseData.responseTime}s` : "Awaiting query"}
          </div>
        </div>

        <div className="nier-content min-h-[100px]"> {/* Added min-height */}
          {loading && (
            <div className="py-12 flex flex-col items-center justify-center">
              <div className="loading-spinner mb-4"></div>
              <p className="text-muted-foreground text-sm uppercase tracking-wider animate-pulse">
                {isDeepResearch ? "Conducting deep research..." : "Retrieving data..."}
              </p>
            </div>
          )}

          {rateLimited && (
             <div className="py-4 nier-container border-destructive">
                <div className="nier-header"><div className="nier-title text-destructive">Rate Limit Exceeded</div></div>
                <div className="nier-content"><p className="text-foreground uppercase tracking-wider text-sm">You've reached the rate limit. Please wait a moment.</p></div>
             </div>
          )}

          {error && !rateLimited && (
            <div className="py-4">
              <p className="text-destructive uppercase tracking-wider">{error}</p>
              {/* Simplified troubleshooting */}
              <div className="mt-4 text-xs text-muted-foreground">
                Please check API server logs for details and ensure it's running and accessible.
              </div>
              <button onClick={() => window.location.reload()} className="mt-4 nier-button text-xs">
                Reload Page
              </button>
            </div>
          )}

          {/* Use MarkdownRenderer for actual content */}
          {!loading && !error && !rateLimited && responseData?.content && (
             <MarkdownRenderer content={responseData.content} />
          )}

          {/* Placeholder when no query/response yet */}
           {!loading && !error && !rateLimited && !responseData && (
             <p className="text-muted-foreground text-sm py-12 text-center">Submit a query to see results.</p>
           )}


          {/* Additional Info Section */}
          {!loading && responseData && responseData.additionalInfo && responseData.additionalInfo.length > 0 && (
            <div className="mt-6 pt-6 border-t border-border">
              <h3 className="text-primary text-sm font-normal uppercase tracking-wider mb-3">System Information</h3>
              <ul className="space-y-2">
                {responseData.additionalInfo.map((info, index) => (
                  <li key={index} className="text-muted-foreground text-xs uppercase tracking-wider">{info}</li>
                ))}
                {/* Display thread ID consistently */}
                {currentThreadId && (
                  <li className="text-muted-foreground text-xs uppercase tracking-wider">Thread ID: {currentThreadId}</li>
                )}
              </ul>
            </div>
          )}
        </div>

        {/* Response Footer (remains the same) */}
        <div className="nier-header border-t-0 border-b-0">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 border border-muted-foreground"></div>
            <span className="text-muted-foreground text-xs uppercase tracking-wider">
              {loading ? "Processing query..." : responseData?.responseTime ? `Response generated in ${responseData.responseTime} seconds` : "Awaiting query"}
            </span>
          </div>
          <div className="text-xs uppercase tracking-wider text-muted-foreground">DELPHOS Search System v1.0</div>
        </div>
      </div>
    </div>
  )
}
