export default function GridBackground() {
  return (
    <div className="absolute inset-0 grid grid-cols-12 grid-rows-12 pointer-events-none">
      {Array.from({ length: 12 }).map((_, i) => (
        <div key={`col-${i}`} className="h-full border-l border-slate-800"></div>
      ))}
      {Array.from({ length: 12 }).map((_, i) => (
        <div key={`row-${i}`} className="w-full border-t border-slate-800"></div>
      ))}
    </div>
  )
}
