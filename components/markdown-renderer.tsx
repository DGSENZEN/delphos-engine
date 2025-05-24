interface MarkdownRendererProps {
  content: string
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  // Parse and render markdown content manually
  const renderMarkdown = (markdown: string) => {
    // Process the markdown content
    let html = markdown

    // Replace code blocks with improved styling
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, language, code) => {
      const lang = language
        ? `<div class="text-xs text-muted-foreground uppercase tracking-wider mb-1">${language}</div>`
        : ""
      return `<pre class="bg-muted p-4 my-4 overflow-x-auto border border-border">${lang}<code class="text-secondary font-mono text-sm">${escapeHtml(code.trim())}</code></pre>`
    })

    // Replace inline code
    html = html.replace(/`([^`]+)`/g, '<code class="bg-muted px-1 py-0.5 text-secondary font-mono">$1</code>')

    // Improve headers with subtle indicators
    html = html.replace(
      /^# (.*$)/gm,
      '<h1 class="text-xl font-normal text-primary uppercase tracking-wider mb-4 mt-6">$1</h1>',
    )
    html = html.replace(
      /^## (.*$)/gm,
      '<h2 class="text-lg font-normal text-primary uppercase tracking-wider mb-3 mt-5">$1</h2>',
    )
    html = html.replace(
      /^### (.*$)/gm,
      '<h3 class="text-base font-normal text-primary uppercase tracking-wider mb-3 mt-4">$1</h3>',
    )
    html = html.replace(
      /^#### (.*$)/gm,
      '<h4 class="text-sm font-normal text-primary uppercase tracking-wider mb-2 mt-4">$1</h4>',
    )

    // Replace bold and italic
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>")
    html = html.replace(/_([^_]+)_/g, "<em>$1</em>")

    // Replace links
    html = html.replace(
      /\[([^\]]+)\]$$([^)]+)$$/g,
      '<a href="$2" class="text-secondary hover:text-secondary/80 underline">$1</a>',
    )

    // Improve list items with subtle animations
    html = html.replace(/^\s*[-*+]\s+(.*)/gm, '<li class="mb-2 list-item-animated">$1</li>')
    html = html.replace(/^\s*\d+\.\s+(.*)/gm, '<li class="mb-2 list-item-animated">$1</li>')

    // Replace unordered lists
    html = html.replace(/<li class="mb-2 list-item-animated">(.*?)<\/li>/gs, (match) => {
      return match
    })

    // Group list items
    let inList = false
    const lines = html.split("\n")
    const result = []

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]
      if (line.includes('<li class="mb-2 list-item-animated">')) {
        if (!inList) {
          result.push('<ul class="list-disc pl-5 mb-4 text-foreground space-y-1">')
          inList = true
        }
        result.push(line)
      } else {
        if (inList) {
          result.push("</ul>")
          inList = false
        }
        result.push(line)
      }
    }

    if (inList) {
      result.push("</ul>")
    }

    html = result.join("\n")

    // Replace blockquotes
    html = html.replace(
      /^>\s+(.*$)/gm,
      '<blockquote class="border-l-4 border-secondary pl-4 italic text-muted-foreground my-4">$1</blockquote>',
    )

    // Replace horizontal rules
    html = html.replace(/^---$/gm, '<hr class="my-6 border-t border-border">')

    // Replace paragraphs (must be done last)
    const paragraphs = html.split(/\n\n+/)
    html = paragraphs
      .map((p) => {
        if (
          p.trim().startsWith("<h") ||
          p.trim().startsWith("<ul") ||
          p.trim().startsWith("<ol") ||
          p.trim().startsWith("<blockquote") ||
          p.trim().startsWith("<pre") ||
          p.trim().startsWith("<hr")
        ) {
          return p
        }
        return `<p class="text-foreground font-light mb-4">${p.replace(/\n/g, "<br>")}</p>`
      })
      .join("\n")

    return html
  }

  // Helper function to escape HTML
  const escapeHtml = (unsafe: string) => {
    return unsafe
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;")
  }

  const renderedContent = renderMarkdown(content)

  return <div className="markdown-content" dangerouslySetInnerHTML={{ __html: renderedContent }} />
}
