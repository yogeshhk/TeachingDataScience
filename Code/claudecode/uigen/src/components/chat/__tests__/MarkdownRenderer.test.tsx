import { test, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { MarkdownRenderer } from "../MarkdownRenderer";

afterEach(() => {
  cleanup();
});

test("renders plain text content", () => {
  render(<MarkdownRenderer content="Hello, world!" />);
  expect(screen.getByText("Hello, world!")).toBeDefined();
});

test("renders markdown heading", () => {
  render(<MarkdownRenderer content="# Hello Heading" />);
  const heading = screen.getByRole("heading", { level: 1 });
  expect(heading).toBeDefined();
  expect(heading.textContent).toBe("Hello Heading");
});

test("renders multiple heading levels", () => {
  const content = `# H1 Heading
## H2 Heading
### H3 Heading`;
  
  render(<MarkdownRenderer content={content} />);
  
  const h1 = screen.getByRole("heading", { level: 1, name: "H1 Heading" });
  const h2 = screen.getByRole("heading", { level: 2, name: "H2 Heading" });
  const h3 = screen.getByRole("heading", { level: 3, name: "H3 Heading" });
  
  expect(h1).toBeDefined();
  expect(h1.tagName).toBe("H1");
  expect(h2).toBeDefined();
  expect(h2.tagName).toBe("H2");
  expect(h3).toBeDefined();
  expect(h3.tagName).toBe("H3");
});

test("renders inline code with custom styling", () => {
  render(<MarkdownRenderer content="Here is `inline code` text" />);
  const codeElement = screen.getByText("inline code");
  expect(codeElement.tagName).toBe("CODE");
  expect(codeElement.className).toContain("not-prose");
  expect(codeElement.className).toContain("text-sm");
  expect(codeElement.className).toContain("px-1");
  expect(codeElement.className).toContain("py-0.5");
  expect(codeElement.className).toContain("rounded-sm");
  expect(codeElement.className).toContain("bg-neutral-100");
  expect(codeElement.className).toContain("text-neutral-900");
  expect(codeElement.className).toContain("font-mono");
});

test("renders code blocks with language class", () => {
  const content = "```javascript\nconst x = 42;\n```";
  render(<MarkdownRenderer content={content} />);
  const codeBlock = screen.getByText("const x = 42;");
  expect(codeBlock.tagName).toBe("CODE");
  expect(codeBlock.className).toContain("language-javascript");
});

test("renders code blocks without custom inline styling", () => {
  const content = "```python\nprint('Hello')\n```";
  render(<MarkdownRenderer content={content} />);
  const codeBlock = screen.getByText("print('Hello')");
  expect(codeBlock.className).not.toContain("not-prose");
  expect(codeBlock.className).not.toContain("bg-gray-100");
});

test("applies custom className to wrapper div", () => {
  const { container } = render(
    <MarkdownRenderer content="Test" className="custom-class" />
  );
  const wrapper = container.firstChild as HTMLElement;
  expect(wrapper.className).toContain("prose");
  expect(wrapper.className).toContain("max-w-none");
  expect(wrapper.className).toContain("custom-class");
});

test("renders bold text", () => {
  render(<MarkdownRenderer content="This is **bold** text" />);
  const boldText = screen.getByText("bold");
  expect(boldText.tagName).toBe("STRONG");
});

test("renders italic text", () => {
  render(<MarkdownRenderer content="This is *italic* text" />);
  const italicText = screen.getByText("italic");
  expect(italicText.tagName).toBe("EM");
});

test("renders links", () => {
  render(<MarkdownRenderer content="[Click here](https://example.com)" />);
  const link = screen.getByRole("link", { name: "Click here" });
  expect(link).toBeDefined();
  expect(link.getAttribute("href")).toBe("https://example.com");
});

test("renders unordered lists", () => {
  const content = `
- Item 1
- Item 2
- Item 3`;
  
  render(<MarkdownRenderer content={content} />);
  const list = screen.getByRole("list");
  expect(list).toBeDefined();
  const items = screen.getAllByRole("listitem");
  expect(items).toHaveLength(3);
  expect(items[0].textContent).toBe("Item 1");
  expect(items[1].textContent).toBe("Item 2");
  expect(items[2].textContent).toBe("Item 3");
});

test("renders ordered lists", () => {
  const content = `
1. First
2. Second
3. Third`;
  
  const { container } = render(<MarkdownRenderer content={content} />);
  const orderedList = container.querySelector("ol");
  expect(orderedList).toBeDefined();
  const items = orderedList?.querySelectorAll("li");
  expect(items).toHaveLength(3);
  expect(items![0].textContent).toBe("First");
  expect(items![1].textContent).toBe("Second");
  expect(items![2].textContent).toBe("Third");
});

test("renders blockquotes", () => {
  render(<MarkdownRenderer content="> This is a quote" />);
  const blockquote = screen.getByText("This is a quote").closest("blockquote");
  expect(blockquote).toBeDefined();
});

test("renders horizontal rules", () => {
  const { container } = render(<MarkdownRenderer content="---" />);
  const hr = container.querySelector("hr");
  expect(hr).toBeDefined();
});

test("renders paragraphs", () => {
  const content = `First paragraph.

Second paragraph.`;
  
  const { container } = render(<MarkdownRenderer content={content} />);
  const paragraphs = container.querySelectorAll("p");
  expect(paragraphs).toHaveLength(2);
  expect(paragraphs[0].textContent).toBe("First paragraph.");
  expect(paragraphs[1].textContent).toBe("Second paragraph.");
});

test("renders tables", () => {
  const content = `| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |`;
  
  // ReactMarkdown by default may not render tables without additional plugins
  // The content will be rendered as plain text in a paragraph
  const { container } = render(<MarkdownRenderer content={content} />);
  const paragraph = container.querySelector("p");
  expect(paragraph).toBeDefined();
  expect(paragraph?.textContent).toContain("Header 1 | Header 2");
});

test("handles empty content", () => {
  const { container } = render(<MarkdownRenderer content="" />);
  const wrapper = container.firstChild as HTMLElement;
  expect(wrapper).toBeDefined();
  expect(wrapper.textContent).toBe("");
});

test("handles complex mixed content", () => {
  const content = `# Title

This is a paragraph with **bold** and *italic* text.

\`\`\`javascript
function hello() {
  return "world";
}
\`\`\`

- List item with \`inline code\`
- Another item

> A blockquote with [a link](https://example.com)`;

  const { container } = render(<MarkdownRenderer content={content} />);
  
  // Verify all elements are rendered
  expect(screen.getByRole("heading", { name: "Title" })).toBeDefined();
  expect(screen.getByText("bold").tagName).toBe("STRONG");
  expect(screen.getByText("italic").tagName).toBe("EM");
  const codeBlock = container.querySelector('code.language-javascript');
  expect(codeBlock).toBeDefined();
  expect(codeBlock?.textContent).toContain('function hello()');
  expect(screen.getByText("inline code")).toBeDefined();
  expect(screen.getByRole("link", { name: "a link" })).toBeDefined();
});

test("preserves code block content without modification", () => {
  const codeContent = `const obj = {
  key: "value",
  nested: {
    deep: true
  }
};`;
  const content = `\`\`\`javascript
${codeContent}
\`\`\``;
  
  const { container } = render(<MarkdownRenderer content={content} />);
  const codeBlock = container.querySelector("code.language-javascript");
  expect(codeBlock).toBeDefined();
  expect(codeBlock?.textContent?.trim()).toBe(codeContent);
});

test("handles code blocks without language specification", () => {
  const content = "```\nplain code block\n```";
  render(<MarkdownRenderer content={content} />);
  const codeBlock = screen.getByText("plain code block");
  expect(codeBlock.tagName).toBe("CODE");
  // Code blocks without language still get custom inline styling in this implementation
  expect(codeBlock.className).toContain("not-prose");
});

test("properly escapes HTML in markdown content", () => {
  const content = "This is <script>alert('xss')</script> text";
  const { container } = render(<MarkdownRenderer content={content} />);
  // ReactMarkdown should escape the script tag
  const paragraph = container.querySelector("p");
  expect(paragraph).toBeDefined();
  expect(paragraph?.innerHTML).toContain("&lt;script&gt;");
  expect(paragraph?.innerHTML).toContain("&lt;/script&gt;");
  // Ensure no actual script tag is rendered
  const scriptTag = container.querySelector("script");
  expect(scriptTag).toBeNull();
});