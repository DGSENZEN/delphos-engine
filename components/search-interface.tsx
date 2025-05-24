"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"
import { Input } from "@/components/ui/input"
import SearchResults from "./search-results"
import { Search, AlertTriangle, Clock, X, Mic, ChevronUp, ChevronDown, FileText, Trash2, BookOpen } from "lucide-react"
import { Button } from "@/components/ui/button"
import WeatherWidget from "./weather-widget"
import { ThemeToggle } from "./theme-toggle"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"

interface SearchHistoryItem {
  id: string
  query: string
  timestamp: number
}

export default function SearchInterface() {
  const [query, setQuery] = useState("")
  const [submittedQuery, setSubmittedQuery] = useState("")
  const [hasSearched, setHasSearched] = useState(false)
  const [scrollY, setScrollY] = useState(0)
  const [viewportHeight, setViewportHeight] = useState(0)
  const [apiStatus, setApiStatus] = useState<"loading" | "online" | "offline" | "mock">("loading")
  const [statusMessage, setStatusMessage] = useState<string>("")
  const [isSearching, setIsSearching] = useState(false)
  const [showPersistentSearch, setShowPersistentSearch] = useState(false)
  const [searchHistory, setSearchHistory] = useState<SearchHistoryItem[]>([])
  const [showHistory, setShowHistory] = useState(false)
  const [historyVisible, setHistoryVisible] = useState(true)
  const [isTransitioning, setIsTransitioning] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [isImageUploading, setIsImageUploading] = useState(false)
  const [currentTime, setCurrentTime] = useState(new Date())
  const [isDeepResearch, setIsDeepResearch] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [threadId, setThreadId] = useState<string | undefined>(undefined)

  const inputRef = useRef<HTMLInputElement>(null)
  const persistentInputRef = useRef<HTMLInputElement>(null)
  const searchFormRef = useRef<HTMLDivElement>(null)
  const resultsRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const debounceTimer = useRef<NodeJS.Timeout>()
  const abortControllerRef = useRef<AbortController | null>(null)

  // Prevent repeated health checks
  const isCheckingHealth = useRef(false)

  // Update current time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)

    return () => clearInterval(timer)
  }, [])

  // Load search history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem("searchHistory")
    if (savedHistory) {
      try {
        const parsedHistory = JSON.parse(savedHistory)
        if (Array.isArray(parsedHistory)) {
          setSearchHistory(parsedHistory)
        }
      } catch (e) {
        console.error("Error parsing search history:", e)
      }
    }

    // Load history visibility preference
    const historyVisibility = localStorage.getItem("historyVisible")
    if (historyVisibility !== null) {
      setHistoryVisible(historyVisibility === "true")
    }

    // Check if we should use mock mode from localStorage
    const useMockMode = localStorage.getItem("useMockResponses") === "true"
    if (useMockMode) {
      setApiStatus("mock")
      setStatusMessage("OFFLINE MODE ACTIVE")
    }
  }, [])

  // Save search history to localStorage when it changes
  useEffect(() => {
    if (searchHistory.length > 0) {
      localStorage.setItem("searchHistory", JSON.stringify(searchHistory))
    }
  }, [searchHistory])

  // Save history visibility preference
  useEffect(() => {
    localStorage.setItem("historyVisible", historyVisible.toString())
  }, [historyVisible])

  // Cleanup function for component unmount
  useEffect(() => {
    return () => {
      // Abort any ongoing request when component unmounts
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current)
      }
    }
  }, [])

  const checkApiHealth = async () => {
    if (isCheckingHealth.current) return
    isCheckingHealth.current = true

    try {
      if (localStorage.getItem("useMockResponses") === "true") {
        setApiStatus("mock")
        setStatusMessage("OFFLINE MODE ACTIVE")
        isCheckingHealth.current = false
        return
      }

      setApiStatus("loading")
      setStatusMessage("CHECKING CONNECTION...")

      try {
        console.log("Checking API health...")
        // Make a single request with proper error handling
        const response = await fetch("/api/health", {
          method: "GET",
          cache: "no-store",
          headers: {
            "Cache-Control": "no-cache",
            Pragma: "no-cache",
          },
        })

        console.log(`Health check response status: ${response.status}`)

        if (response.ok) {
          try {
            const data = await response.json()
            console.log("Health check response data:", data)
            setApiStatus(data.status === "ok" ? "online" : "offline")
            setStatusMessage(data.status === "ok" ? "SYSTEM ONLINE" : `API STATUS: ${data.status}`)
          } catch (jsonError) {
            console.error("Error parsing health check JSON:", jsonError)
            // If we can't parse JSON but the response was OK, we'll still consider it online
            setApiStatus("online")
            setStatusMessage("SYSTEM ONLINE (NON-JSON RESPONSE)")
          }
        } else {
          console.error(`Health check failed with status: ${response.status}`)
          setApiStatus("offline")
          setStatusMessage(`CONNECTION ERROR: ${response.status}`)

          // Offer to switch to mock mode
          setTimeout(() => {
            if (confirm("API connection failed. Would you like to switch to offline mode?")) {
              localStorage.setItem("useMockResponses", "true")
              setApiStatus("mock")
              setStatusMessage("OFFLINE MODE ACTIVE")
            }
          }, 500)
        }
      } catch (error) {
        console.error("Health check fetch error:", error)
        setApiStatus("offline")
        setStatusMessage("API UNREACHABLE")

        // Offer to switch to mock mode
        setTimeout(() => {
          if (confirm("API connection failed. Would you like to switch to offline mode?")) {
            localStorage.setItem("useMockResponses", "true")
            setApiStatus("mock")
            setStatusMessage("OFFLINE MODE ACTIVE")
          }
        }, 500)
      }
    } finally {
      isCheckingHealth.current = false
    }
  }

  // Toggle between mock and online mode
  const toggleMockMode = () => {
    if (apiStatus === "mock") {
      localStorage.removeItem("useMockResponses")
      setApiStatus("loading")
      setStatusMessage("CHECKING CONNECTION...")

      // Re-check API health
      setTimeout(() => {
        checkApiHealth()
      }, 500)
    } else {
      localStorage.setItem("useMockResponses", "true")
      setApiStatus("mock")
      setStatusMessage("OFFLINE MODE ACTIVE")
    }
  }

  // Add this debugging function after the toggleMockMode function
  const debugApiConnection = async () => {
    try {
      setApiStatus("loading")
      setStatusMessage("TESTING CONNECTION...")

      // Test health endpoint
      const healthResponse = await fetch("/api/health", {
        method: "GET",
        cache: "no-store",
      })

      console.log("Health check response:", healthResponse.status)
      const healthData = await healthResponse.json()
      console.log("Health check data:", healthData)

      // Test a simple query with minimal data
      const testFormData = new FormData()
      testFormData.append("query", "test query")

      const searchResponse = await fetch("/api/search", {
        method: "POST",
        body: testFormData,
      })

      console.log("Test search response status:", searchResponse.status)
      const searchText = await searchResponse.text()
      console.log("Test search response text:", searchText)

      try {
        const searchData = JSON.parse(searchText)
        console.log("Test search data:", searchData)
        alert("API test completed. Check console for details.")
      } catch (e) {
        console.error("Failed to parse search response as JSON:", e)
        alert("API test completed with errors. Check console for details.")
      }

      // Update status based on results
      setApiStatus("online")
      setStatusMessage("CONNECTION TEST COMPLETE")
    } catch (error) {
      console.error("API test error:", error)
      setApiStatus("offline")
      setStatusMessage("CONNECTION TEST FAILED")
      alert(`API test failed: ${error.message}`)
    }
  }

  // Add query to search history
  const addToSearchHistory = (query: string) => {
    if (!query.trim()) return

    // Create new history item
    const newItem = {
      id: Date.now().toString(),
      query: query.trim(),
      timestamp: Date.now(),
    }

    // Remove duplicates and limit to 10 items
    const filteredHistory = searchHistory.filter((item) => item.query !== query.trim()).slice(0, 9)

    // Add new item at the beginning
    setSearchHistory([newItem, ...filteredHistory])
  }

  // Clear search history
  const clearSearchHistory = () => {
    setSearchHistory([])
    localStorage.removeItem("searchHistory")
  }

  // Toggle history visibility
  const toggleHistoryVisibility = () => {
    setHistoryVisible(!historyVisible)
  }

  // Handle file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files)

      // Check file size limits (10MB per file, 20MB total)
      const maxFileSize = 10 * 1024 * 1024 // 10MB
      const maxTotalSize = 20 * 1024 * 1024 // 20MB

      let totalSize = uploadedFiles.reduce((sum, file) => sum + file.size, 0)

      const validFiles = newFiles.filter((file) => {
        if (file.size > maxFileSize) {
          alert(`File ${file.name} exceeds the 10MB size limit.`)
          return false
        }

        totalSize += file.size
        if (totalSize > maxTotalSize) {
          alert(`Total upload size exceeds the 20MB limit.`)
          return false
        }

        return true
      })

      setUploadedFiles((prev) => [...prev, ...validFiles])
    }

    // Reset the input value so the same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  // Remove a file from the uploaded files
  const removeFile = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index))
  }

  // Clear all uploaded files
  const clearFiles = () => {
    setUploadedFiles([])
  }

  // Handle search submission with smooth transition
  const handleSearchSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (isSearching || !query.trim()) return

    // Abort any previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    setIsSearching(true)
    setIsTransitioning(true)

    // Add to search history
    addToSearchHistory(query)

    // Hide history dropdown
    setShowHistory(false)

    // Smooth transition to results view
    if (!hasSearched) {
      // First search - animate to results view
      window.scrollTo({
        top: 0,
        behavior: "smooth",
      })

      // Delay setting hasSearched to allow for smooth animation
      setTimeout(() => {
        setHasSearched(true)
        setSubmittedQuery(query)
        setShowPersistentSearch(true)
        setIsTransitioning(false)

        // Scroll to results after a short delay to ensure they're rendered
        setTimeout(() => {
          if (resultsRef.current) {
            const topOffset = resultsRef.current.offsetTop - 80 // Subtract header height
            window.scrollTo({
              top: topOffset,
              behavior: "smooth",
            })
          }
        }, 100)
      }, 600)
    } else {
      // Subsequent searches - just update the query
      setSubmittedQuery(query)

      // Scroll to top of results
      setTimeout(() => {
        if (resultsRef.current) {
          const topOffset = resultsRef.current.offsetTop - 80 // Subtract header height
          window.scrollTo({
            top: topOffset,
          })
        }
        setIsTransitioning(false)
      }, 300)
    }

    // Reset isSearching after a delay to prevent rapid submissions
    setTimeout(() => {
      setIsSearching(false)
    }, 1000)
  }

  // Select a query from history
  const selectHistoryItem = (query: string) => {
    setQuery(query)
    setShowHistory(false)

    // Focus the appropriate input
    if (showPersistentSearch && persistentInputRef.current) {
      persistentInputRef.current.focus()
    } else if (inputRef.current) {
      inputRef.current.focus()
    }
  }

  // Handle audio input
  const handleAudioInput = () => {
    // Check if browser supports speech recognition
    if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
      const recognition = new SpeechRecognition()

      recognition.lang = "en-US"
      recognition.interimResults = false

      setIsRecording(true)

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript
        setQuery(transcript)
        setIsRecording(false)

        // Focus the appropriate input
        if (showPersistentSearch && persistentInputRef.current) {
          persistentInputRef.current.focus()
        } else if (inputRef.current) {
          inputRef.current.focus()
        }
      }

      recognition.onend = () => {
        setIsRecording(false)
      }

      recognition.onerror = () => {
        setIsRecording(false)
        alert("Error occurred during voice recognition. Please try again.")
      }

      recognition.start()
    } else {
      alert("Speech recognition is not supported in your browser. Try Chrome or Edge.")
    }
  }

  // Handle image input
  const handleImageInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click()
    }
  }

  // Track scroll position and viewport dimensions
  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY)

      // Show persistent search bar when scrolled past the hero section
      // or when a search has been performed
      if (hasSearched || window.scrollY > viewportHeight * 0.3) {
        setShowPersistentSearch(true)
      } else {
        setShowPersistentSearch(false)
      }
    }

    const handleResize = () => {
      setViewportHeight(window.innerHeight)
    }

    handleResize()
    handleScroll()

    window.addEventListener("scroll", handleScroll, { passive: true })
    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("scroll", handleScroll)
      window.removeEventListener("resize", handleResize)
    }
  }, [viewportHeight, hasSearched])

  // Focus input on initial load
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus()
    }
  }, [])

  // Format current time in NieR: Automata style
  const formatTime = (date: Date) => {
    const hours = date.getHours().toString().padStart(2, "0")
    const minutes = date.getMinutes().toString().padStart(2, "0")
    const seconds = date.getSeconds().toString().padStart(2, "0")
    return `${hours}:${minutes}:${seconds}`
  }

  // Format current date in NieR: Automata style
  const formatDate = (date: Date) => {
    const day = date.getDate().toString().padStart(2, "0")
    const month = (date.getMonth() + 1).toString().padStart(2, "0")
    const year = date.getFullYear()
    return `${day}.${month}.${year}`
  }

  // Handle click outside to close history dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (showHistory && searchFormRef.current && !searchFormRef.current.contains(event.target as Node)) {
        setShowHistory(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [showHistory])

  return (
    <div className="min-h-screen font-mono overflow-x-hidden">
      {/* Top navigation bar */}
      <div className="nier-container border-b sticky top-0 z-50">
        <div className="flex items-center justify-between px-4 py-2">
          <div className="flex items-center gap-4">
            <div className="nier-title">DELPHOS SEARCH SYSTEM</div>
            <div className="nier-dots-border h-6 w-px mx-2"></div>
            <div className="system-status">
              <div
                className={`w-2 h-2 rounded-full status-indicator-dot ${
                  apiStatus === "loading"
                    ? "bg-yellow-500"
                    : apiStatus === "online"
                      ? "bg-green-500"
                      : apiStatus === "mock"
                        ? "bg-accent"
                        : "bg-red-500"
                }`}
              ></div>
              <span className="typing-cursor">{statusMessage}</span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <ThemeToggle />
            <button onClick={toggleMockMode} className="nier-action-button text-xs">
              {apiStatus === "mock" ? "CONNECT" : "OFFLINE MODE"}
            </button>
            <button onClick={debugApiConnection} className="nier-action-button text-xs">
              TEST API
            </button>
          </div>
        </div>
      </div>

      {/* Persistent Search Bar (visible when scrolled or after search) */}
      {showPersistentSearch && (
        <div className="nier-container border-b sticky top-12 z-40 transition-all duration-300">
          <div className="max-w-3xl mx-auto px-4 py-3">
            <form onSubmit={handleSearchSubmit} className="relative">
              <div className="relative">
                <Input
                  ref={persistentInputRef}
                  type="text"
                  placeholder={isDeepResearch ? "ENTER RESEARCH TOPIC..." : "ENTER QUERY..."}
                  className="nier-input pr-36 font-mono uppercase text-sm"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onFocus={() => historyVisible && setShowHistory(true)}
                  onDoubleClick={() => historyVisible && setShowHistory(true)}
                />
                <div className="absolute right-0 top-0 h-full flex items-center gap-1 pr-1">
                  <button
                    type="button"
                    className={`nier-action-button h-8 w-8 flex items-center justify-center ${
                      isRecording ? "bg-accent text-accent-foreground" : ""
                    }`}
                    onClick={handleAudioInput}
                    disabled={isRecording || isImageUploading}
                    title="Voice input"
                  >
                    <Mic size={16} />
                  </button>
                  <button
                    type="button"
                    className={`nier-action-button h-8 w-8 flex items-center justify-center ${
                      isImageUploading ? "bg-accent text-accent-foreground" : ""
                    }`}
                    onClick={handleImageInput}
                    disabled={isRecording || isImageUploading}
                    title="Upload files"
                  >
                    <FileText size={16} />
                  </button>
                  <button
                    type="submit"
                    className="nier-button h-8 flex items-center justify-center"
                    disabled={isSearching || !query.trim()}
                    aria-label="Search"
                  >
                    {isSearching ? (
                      <div className="h-4 w-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin"></div>
                    ) : (
                      <Search size={16} />
                    )}
                  </button>
                </div>
              </div>

              {/* Search History Dropdown */}
              {showHistory && searchHistory.length > 0 && historyVisible && (
                <div className="absolute top-full left-0 right-0 mt-1 nier-container z-50">
                  <div className="nier-header">
                    <div className="flex items-center text-xs">
                      <Clock size={12} className="mr-1" />
                      <span className="uppercase tracking-wider">Query History</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={(e) => {
                          e.preventDefault()
                          setShowHistory(false)
                        }}
                        className="text-xs text-muted-foreground hover:text-accent"
                      >
                        <X size={12} />
                      </button>
                      <button
                        onClick={(e) => {
                          e.preventDefault()
                          clearSearchHistory()
                        }}
                        className="text-xs uppercase tracking-wider text-muted-foreground hover:text-accent"
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                  <div className="max-h-60 overflow-y-auto">
                    {searchHistory.map((item) => (
                      <div
                        key={item.id}
                        className={`search-history-item uppercase ${item.query === query ? "active" : ""}`}
                        onClick={() => selectHistoryItem(item.query)}
                      >
                        {item.query}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </form>
          </div>
        </div>
      )}

      {/* Main container */}
      <div className={`relative ${isTransitioning ? "transition-all duration-500" : ""}`}>
        {/* Hero section */}
        <section
          className={`min-h-screen flex flex-col items-center justify-center px-6 relative ${hasSearched ? "pt-20" : ""}`}
        >
          {/* Weather Widget - Top Right Corner */}
          <div
            className={`absolute top-4 right-4 z-10 transition-all duration-500 ${hasSearched ? "opacity-0 pointer-events-none" : ""}`}
          >
            <WeatherWidget compact={true} />
          </div>

          {/* Weather Widget - After Search (Fixed Position) */}
          {hasSearched && (
            <div className="fixed bottom-4 right-4 z-50 transition-all duration-500">
              <WeatherWidget compact={true} />
            </div>
          )}

          {/* Header */}
          {!hasSearched && (
            <header className="relative z-10 text-center mb-12 pt-12">
              <div className="flex flex-col items-center">
                <div className="w-20 h-20 border-2 border-accent mb-8 relative logo-container">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-2xl font-light">DELPHOS</span>
                  </div>
                  <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-accent logo-corner"></div>
                  <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-accent logo-corner"></div>
                  <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-accent logo-corner"></div>
                  <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-accent logo-corner"></div>
                </div>
                <h1 className="text-4xl font-light tracking-widest mb-4 uppercase">Search System</h1>
                <div className="flex items-center gap-4">
                  <div className="h-px w-16 bg-border"></div>
                  <p className="text-sm text-muted-foreground tracking-wider uppercase">Data Retrieval Interface</p>
                  <div className="h-px w-16 bg-border"></div>
                </div>
              </div>
            </header>
          )}

          {/* Search form */}
          <div
            ref={searchFormRef}
            className={`relative z-10 w-full max-w-3xl mx-auto px-4 transition-all duration-500 ${
              hasSearched ? "opacity-0 pointer-events-none" : "opacity-100"
            }`}
          >
            {apiStatus === "offline" && (
              <div className="mb-4 nier-container p-3 border-destructive">
                <div className="flex items-center gap-3">
                  <AlertTriangle size={18} className="text-destructive" />
                  <p className="text-sm uppercase tracking-wider glitch-text">Connection error: API unavailable</p>
                </div>
              </div>
            )}

            {apiStatus === "mock" && (
              <div className="mb-4 nier-container p-3 border-accent">
                <div className="flex items-center gap-3">
                  <AlertTriangle size={18} className="text-accent" />
                  <p className="text-sm uppercase tracking-wider">
                    Operating in offline mode. Responses are simulated and not from the actual API.
                  </p>
                </div>
              </div>
            )}

            <form onSubmit={handleSearchSubmit} className="mb-6">
              <div className="relative">
                <div className="nier-container">
                  <div className="nier-header">
                    <div className="flex items-center gap-2">
                      <span className="text-sm uppercase tracking-wider">
                        {isDeepResearch ? "Deep Research Mode" : "Standard Query Mode"}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Switch id="research-mode" checked={isDeepResearch} onCheckedChange={setIsDeepResearch} />
                      <Label htmlFor="research-mode" className="text-xs uppercase tracking-wider cursor-pointer">
                        <BookOpen className="w-3 h-3 inline-block mr-1" />
                        Deep Research
                      </Label>
                    </div>
                  </div>
                  <div className="relative p-2">
                    <Input
                      id="search-query"
                      ref={inputRef}
                      type="text"
                      placeholder={isDeepResearch ? "ENTER RESEARCH TOPIC..." : "ENTER QUERY..."}
                      className="nier-input pr-36 font-mono uppercase text-sm"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      onFocus={() => historyVisible && setShowHistory(true)}
                      onDoubleClick={() => historyVisible && setShowHistory(true)}
                    />
                    <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex items-center gap-1">
                      <button
                        type="button"
                        className={`nier-action-button h-8 w-8 flex items-center justify-center ${
                          isRecording ? "bg-accent text-accent-foreground" : ""
                        }`}
                        onClick={handleAudioInput}
                        disabled={isRecording || isImageUploading}
                        title="Voice input"
                      >
                        <Mic size={16} />
                      </button>
                      <button
                        type="button"
                        className={`nier-action-button h-8 w-8 flex items-center justify-center ${
                          uploadedFiles.length > 0 ? "bg-accent text-accent-foreground" : ""
                        }`}
                        onClick={handleImageInput}
                        disabled={isRecording || isImageUploading}
                        title="Upload files"
                      >
                        <FileText size={16} />
                      </button>
                      <button
                        type="submit"
                        className="nier-button h-8 flex items-center justify-center"
                        disabled={isSearching || !query.trim()}
                        aria-label="Search"
                      >
                        {isSearching ? (
                          <div className="h-4 w-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin"></div>
                        ) : (
                          <Search size={16} />
                        )}
                      </button>
                    </div>

                    {/* Hidden file input */}
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      onChange={handleFileUpload}
                      className="hidden"
                      accept="image/jpeg,image/png,image/webp,image/gif,audio/mpeg,audio/wav,audio/ogg,audio/mp4,video/mp4,video/quicktime,video/webm,video/mpeg"
                    />
                  </div>

                  {/* Uploaded files display */}
                  {uploadedFiles.length > 0 && (
                    <div className="p-2 border-t border-border">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-xs uppercase tracking-wider">
                          Uploaded Files ({uploadedFiles.length})
                        </span>
                        <button
                          type="button"
                          onClick={clearFiles}
                          className="text-xs text-muted-foreground hover:text-accent flex items-center gap-1"
                        >
                          <Trash2 size={12} />
                          <span>Clear All</span>
                        </button>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {uploadedFiles.map((file, index) => (
                          <div key={index} className="flex items-center gap-1 bg-muted px-2 py-1 rounded text-xs">
                            <span className="max-w-[150px] truncate">{file.name}</span>
                            <span className="text-muted-foreground">({(file.size / 1024).toFixed(1)} KB)</span>
                            <button
                              type="button"
                              onClick={() => removeFile(index)}
                              className="text-muted-foreground hover:text-accent ml-1"
                            >
                              <X size={12} />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Search History Dropdown */}
                {showHistory && searchHistory.length > 0 && historyVisible && (
                  <div className="absolute top-full left-0 right-0 mt-1 nier-container z-50">
                    <div className="nier-header">
                      <div className="flex items-center text-xs">
                        <Clock size={12} className="mr-1" />
                        <span className="uppercase tracking-wider">Query History</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={(e) => {
                            e.preventDefault()
                            setShowHistory(false)
                          }}
                          className="text-xs text-muted-foreground hover:text-accent"
                        >
                          <X size={12} />
                        </button>
                        <button
                          onClick={(e) => {
                            e.preventDefault()
                            clearSearchHistory()
                          }}
                          className="text-xs uppercase tracking-wider text-muted-foreground hover:text-accent"
                        >
                          Clear
                        </button>
                      </div>
                    </div>
                    <div className="max-h-60 overflow-y-auto">
                      {searchHistory.map((item) => (
                        <div
                          key={item.id}
                          className={`search-history-item uppercase ${item.query === query ? "active" : ""}`}
                          onClick={() => selectHistoryItem(item.query)}
                        >
                          {item.query}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="mt-4 text-center text-xs text-muted-foreground uppercase tracking-wider">
                  {apiStatus === "offline"
                    ? "API is currently offline. Switch to offline mode to use simulated responses."
                    : apiStatus === "mock"
                      ? "Using simulated responses in offline mode."
                      : isDeepResearch
                        ? "Deep research mode provides comprehensive analysis (3 requests/minute limit)"
                        : "DELPHOS search system ready for query input (10 requests/minute limit)"}
                </div>
              </div>
            </form>

            {/* History hint */}
            <div className="flex justify-center items-center gap-4 text-xs text-muted-foreground mt-2 opacity-70">
              <span className="inline-flex items-center uppercase tracking-wider">
                <span className="mr-1">※</span>
                Double-click input field to view query history
              </span>

              <Button
                onClick={toggleHistoryVisibility}
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs text-muted-foreground hover:text-accent uppercase tracking-wider"
              >
                {historyVisible ? (
                  <span className="flex items-center">
                    <ChevronUp size={14} className="mr-1" />
                    Hide History
                  </span>
                ) : (
                  <span className="flex items-center">
                    <ChevronDown size={14} className="mr-1" />
                    Show History
                  </span>
                )}
              </Button>
            </div>
          </div>
        </section>

        {/* Results section */}
        {hasSearched && (
          <section
            ref={resultsRef}
            className="min-h-screen pt-16 pb-24 px-6 relative z-10 flex flex-col items-center results-section"
          >
            <SearchResults
              query={submittedQuery}
              files={uploadedFiles}
              useMockResponse={apiStatus === "mock"}
              isDeepResearch={isDeepResearch}
              threadId={threadId}
            />
          </section>
        )}

        {/* Footer */}
        <footer className="relative z-10 py-6 px-6 nier-dots-border">
          <div className="max-w-5xl mx-auto">
            <div className="flex flex-col md:flex-row justify-between items-center gap-6">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 border border-muted-foreground"></div>
                <p className="text-muted-foreground text-xs tracking-wider uppercase">
                  DELPHOS SEARCH SYSTEM © {new Date().getFullYear()}
                </p>
              </div>

              <div className="nier-menu">
                <a href="#" className="nier-menu-item">
                  About
                </a>
                <a href="#" className="nier-menu-item">
                  Privacy
                </a>
                <a href="#" className="nier-menu-item">
                  Terms
                </a>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  )
}
