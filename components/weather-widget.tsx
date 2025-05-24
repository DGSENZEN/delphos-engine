"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import {
  Cloud,
  CloudRain,
  CloudSnow,
  Sun,
  CloudLightning,
  CloudFog,
  Wind,
  Loader2,
  ChevronDown,
  ChevronUp,
  Search,
  MapPin,
} from "lucide-react"
import { Input } from "@/components/ui/input"

interface WeatherData {
  location: string
  temperature: number
  condition: string
  icon: string
  humidity: number
  windSpeed: number
  feelsLike: number
  lastUpdated: string
  country: string
}

interface WeatherWidgetProps {
  className?: string
  compact?: boolean
}

export default function WeatherWidget({ className = "", compact = false }: WeatherWidgetProps) {
  const [weather, setWeather] = useState<WeatherData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)
  const [searchMode, setSearchMode] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [searchLoading, setSearchLoading] = useState(false)
  const searchInputRef = useRef<HTMLInputElement>(null)

  // Get weather by coordinates
  const fetchWeatherByCoords = async (lat: number, lon: number) => {
    try {
      setLoading(true)
      const apiKey = process.env.NEXT_PUBLIC_OPENWEATHER_API_KEY

      if (!apiKey) {
        setError("API key not configured")
        setLoading(false)
        return
      }

      const response = await fetch(
        `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric`,
      )

      if (!response.ok) {
        throw new Error(`Weather API error: ${response.status}`)
      }

      const data = await response.json()

      setWeather({
        location: data.name,
        temperature: Math.round(data.main.temp),
        condition: data.weather[0].main,
        icon: data.weather[0].icon,
        humidity: data.main.humidity,
        windSpeed: data.wind.speed,
        feelsLike: Math.round(data.main.feels_like),
        lastUpdated: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        country: data.sys.country,
      })
      setLoading(false)
    } catch (err) {
      console.error("Error fetching weather:", err)
      setError("Failed to fetch weather")
      setLoading(false)
    }
  }

  // Get weather by city name
  const fetchWeatherByCity = async (city: string) => {
    try {
      setLoading(true)
      const apiKey = process.env.NEXT_PUBLIC_OPENWEATHER_API_KEY

      if (!apiKey) {
        setError("API key not configured")
        setLoading(false)
        return
      }

      const response = await fetch(
        `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${apiKey}&units=metric`,
      )

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error("Location not found")
        }
        throw new Error(`Weather API error: ${response.status}`)
      }

      const data = await response.json()

      setWeather({
        location: data.name,
        temperature: Math.round(data.main.temp),
        condition: data.weather[0].main,
        icon: data.weather[0].icon,
        humidity: data.main.humidity,
        windSpeed: data.wind.speed,
        feelsLike: Math.round(data.main.feels_like),
        lastUpdated: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        country: data.sys.country,
      })
      setLoading(false)
      // Exit search mode after successful search
      setSearchMode(false)
      setSearchQuery("")
    } catch (err: any) {
      console.error("Error fetching weather:", err)
      setError(err.message || "Failed to fetch weather")
      setLoading(false)
    }
  }

  // Search for cities
  const searchCities = async (query: string) => {
    if (!query.trim()) {
      setSearchResults([])
      return
    }

    try {
      setSearchLoading(true)
      const apiKey = process.env.NEXT_PUBLIC_OPENWEATHER_API_KEY

      if (!apiKey) {
        setError("API key not configured")
        setSearchLoading(false)
        return
      }

      const response = await fetch(`https://api.openweathermap.org/geo/1.0/direct?q=${query}&limit=5&appid=${apiKey}`)

      if (!response.ok) {
        throw new Error(`Geocoding API error: ${response.status}`)
      }

      const data = await response.json()
      setSearchResults(data)
      setSearchLoading(false)
    } catch (err) {
      console.error("Error searching cities:", err)
      setSearchLoading(false)
      setSearchResults([])
    }
  }

  // Handle search input change with debounce
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setSearchQuery(value)

    // Debounce search
    const timeoutId = setTimeout(() => {
      searchCities(value)
    }, 500)

    return () => clearTimeout(timeoutId)
  }

  // Select a city from search results
  const selectCity = (city: any) => {
    fetchWeatherByCity(`${city.name},${city.country}`)
  }

  // Initialize with user's location
  useEffect(() => {
    const getLocation = () => {
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            const { latitude, longitude } = position.coords
            fetchWeatherByCoords(latitude, longitude)
          },
          (err) => {
            console.error("Error getting location:", err)
            setError("Location access denied")
            setLoading(false)
          },
          { timeout: 10000 },
        )
      } else {
        setError("Geolocation not supported")
        setLoading(false)
      }
    }

    getLocation()

    // Refresh weather data every 30 minutes
    const refreshInterval = setInterval(
      () => {
        if (weather?.location) {
          fetchWeatherByCity(`${weather.location},${weather.country}`)
        } else {
          getLocation()
        }
      },
      30 * 60 * 1000,
    )

    return () => clearInterval(refreshInterval)
  }, [])

  // Focus search input when entering search mode
  useEffect(() => {
    if (searchMode && searchInputRef.current) {
      searchInputRef.current.focus()
    }
  }, [searchMode])

  // Function to render weather icon based on condition
  const renderWeatherIcon = () => {
    if (!weather) return <Cloud className="w-4 h-4 text-accent" />

    const iconCode = weather.icon
    const isNight = iconCode.includes("n")
    const iconSize = compact ? 4 : 5

    switch (weather.condition.toLowerCase()) {
      case "clear":
        return isNight ? (
          <Sun className={`w-${iconSize} h-${iconSize} text-accent`} />
        ) : (
          <Sun className={`w-${iconSize} h-${iconSize} text-accent`} />
        )
      case "clouds":
        return <Cloud className={`w-${iconSize} h-${iconSize} text-accent`} />
      case "rain":
      case "drizzle":
        return <CloudRain className={`w-${iconSize} h-${iconSize} text-accent`} />
      case "snow":
        return <CloudSnow className={`w-${iconSize} h-${iconSize} text-accent`} />
      case "thunderstorm":
        return <CloudLightning className={`w-${iconSize} h-${iconSize} text-accent`} />
      case "mist":
      case "fog":
      case "haze":
        return <CloudFog className={`w-${iconSize} h-${iconSize} text-accent`} />
      default:
        return <Cloud className={`w-${iconSize} h-${iconSize} text-accent`} />
    }
  }

  // Toggle expanded view
  const toggleExpanded = () => {
    if (searchMode) {
      setSearchMode(false)
      setSearchQuery("")
      setSearchResults([])
    } else {
      setExpanded(!expanded)
    }
  }

  // Enter search mode
  const enterSearchMode = (e: React.MouseEvent) => {
    e.stopPropagation()
    setSearchMode(true)
    setExpanded(true)
  }

  // Handle search form submission
  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchQuery.trim()) {
      fetchWeatherByCity(searchQuery)
    }
  }

  // Compact view (non-expanded)
  if (compact && !expanded) {
    return (
      <div
        className={`nier-container cursor-pointer transition-all duration-300 weather-widget ${className}`}
        onClick={toggleExpanded}
      >
        <div className="nier-header py-1 px-2">
          <div className="flex items-center gap-1">
            <span className="text-xs uppercase tracking-wider">WEATHER</span>
          </div>
          <div className="text-xs uppercase tracking-wider">{weather?.lastUpdated || "--:--"}</div>
        </div>
        <div className="p-2 flex items-center justify-between gap-3">
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin text-accent" />
          ) : error ? (
            <div className="text-xs text-destructive flex items-center gap-1">
              <span>ERROR</span>
            </div>
          ) : weather ? (
            <>
              <div className="flex items-center gap-2">
                {renderWeatherIcon()}
                <span className="text-sm font-light">{weather.temperature}°C</span>
              </div>
              <div className="text-xs uppercase tracking-wider text-muted-foreground">{weather.condition}</div>
            </>
          ) : null}
        </div>
      </div>
    )
  }

  // Full view (expanded)
  return (
    <div
      className={`nier-container transition-all duration-300 weather-widget ${className} ${expanded ? "w-64" : "w-auto"}`}
    >
      <div className="nier-header cursor-pointer" onClick={toggleExpanded}>
        <div className="nier-title flex items-center gap-2 text-xs">
          <span>WEATHER</span>
        </div>
        <div className="text-xs uppercase tracking-wider flex items-center gap-1">
          {weather?.lastUpdated || "--:--"}
          {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </div>
      </div>

      <div className="nier-content">
        {searchMode ? (
          <div className="p-2 space-y-2">
            <form onSubmit={handleSearchSubmit}>
              <div className="relative">
                <Input
                  ref={searchInputRef}
                  type="text"
                  placeholder="SEARCH LOCATION..."
                  className="nier-input pr-8 font-mono uppercase text-xs"
                  value={searchQuery}
                  onChange={handleSearchChange}
                />
                <button
                  type="submit"
                  className="absolute right-2 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-accent"
                  disabled={searchLoading}
                >
                  {searchLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                </button>
              </div>
            </form>

            {searchResults.length > 0 && (
              <div className="max-h-40 overflow-y-auto border border-border">
                {searchResults.map((city, index) => (
                  <div
                    key={`${city.name}-${city.country}-${index}`}
                    className="p-2 text-xs uppercase hover:bg-muted cursor-pointer flex items-center justify-between"
                    onClick={() => selectCity(city)}
                  >
                    <div className="flex items-center gap-1">
                      <MapPin className="w-3 h-3" />
                      <span>{city.name}</span>
                    </div>
                    <span className="text-muted-foreground">{city.country}</span>
                  </div>
                ))}
              </div>
            )}

            {searchQuery && searchResults.length === 0 && !searchLoading && (
              <div className="text-xs text-muted-foreground p-1">No locations found</div>
            )}
          </div>
        ) : loading ? (
          <div className="flex items-center justify-center py-2">
            <Loader2 className="w-5 h-5 animate-spin text-accent" />
          </div>
        ) : error ? (
          <div className="text-xs text-destructive uppercase tracking-wider py-2">{error}</div>
        ) : weather ? (
          <div className="space-y-2 p-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="weather-icon-container">{renderWeatherIcon()}</div>
                <span className="text-lg font-light">{weather.temperature}°C</span>
              </div>
              <div className="text-xs uppercase tracking-wider text-muted-foreground">{weather.condition}</div>
            </div>

            <div className="pt-2 space-y-2 border-t border-border animate-fadeIn">
              <div className="flex items-center justify-between">
                <div className="text-xs uppercase tracking-wider">
                  {weather.location}, {weather.country}
                </div>
                <button
                  onClick={enterSearchMode}
                  className="text-xs text-muted-foreground hover:text-accent flex items-center gap-1"
                >
                  <Search className="w-3 h-3" />
                  <span>CHANGE</span>
                </button>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center gap-1">
                  <span className="text-muted-foreground">Feels:</span>
                  <span>{weather.feelsLike}°C</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-muted-foreground">Humidity:</span>
                  <span>{weather.humidity}%</span>
                </div>
                <div className="flex items-center gap-1 col-span-2">
                  <Wind className="w-3 h-3 text-muted-foreground" />
                  <span>{weather.windSpeed} m/s</span>
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
