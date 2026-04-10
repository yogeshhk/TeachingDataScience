import { test, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { MessageList } from "../MessageList";
import type { Message } from "ai";

// Mock the MarkdownRenderer component
vi.mock("../MarkdownRenderer", () => ({
  MarkdownRenderer: ({ content }: { content: string }) => <div>{content}</div>,
}));

afterEach(() => {
  cleanup();
});

test("MessageList shows empty state when no messages", () => {
  render(<MessageList messages={[]} />);

  expect(
    screen.getByText("Start a conversation to generate React components")
  ).toBeDefined();
  expect(
    screen.getByText("I can help you create buttons, forms, cards, and more")
  ).toBeDefined();
});

test("MessageList renders user messages", () => {
  const messages: Message[] = [
    {
      id: "1",
      role: "user",
      content: "Create a button component",
    },
  ];

  render(<MessageList messages={messages} />);

  expect(screen.getByText("Create a button component")).toBeDefined();
});

test("MessageList renders assistant messages", () => {
  const messages: Message[] = [
    {
      id: "1",
      role: "assistant",
      content: "I'll help you create a button component.",
    },
  ];

  render(<MessageList messages={messages} />);

  expect(
    screen.getByText("I'll help you create a button component.")
  ).toBeDefined();
});

test("MessageList renders messages with parts", () => {
  const messages: Message[] = [
    {
      id: "1",
      role: "assistant",
      content: "",
      parts: [
        { type: "text", text: "Creating your component..." },
        {
          type: "tool-invocation",
          toolInvocation: {
            toolCallId: "asdf",
            args: {},
            toolName: "str_replace_editor",
            state: "result",
            result: "Success",
          },
        },
      ],
    },
  ];

  render(<MessageList messages={messages} />);

  expect(screen.getByText("Creating your component...")).toBeDefined();
  expect(screen.getByText("str_replace_editor")).toBeDefined();
});

test("MessageList shows content for assistant message with content", () => {
  const messages: Message[] = [
    {
      id: "1",
      role: "assistant",
      content: "Generating your component...",
    },
  ];

  render(<MessageList messages={messages} isLoading={true} />);

  // The component shows the content but not a loading indicator when content is present
  expect(screen.getByText("Generating your component...")).toBeDefined();
  expect(screen.queryByText("Generating...")).toBeNull();
});

test("MessageList shows loading state for last assistant message without content", () => {
  const messages: Message[] = [
    {
      id: "1",
      role: "assistant",
      content: "",
    },
  ];

  render(<MessageList messages={messages} isLoading={true} />);

  expect(screen.getByText("Generating...")).toBeDefined();
});

test("MessageList doesn't show loading state for non-last messages", () => {
  const messages: Message[] = [
    {
      id: "1",
      role: "assistant",
      content: "First response",
    },
    {
      id: "2",
      role: "user",
      content: "Another request",
    },
  ];

  render(<MessageList messages={messages} isLoading={true} />);

  // Loading state should not appear because the last message is from user, not assistant
  expect(screen.queryByText("Generating...")).toBeNull();
});

test("MessageList renders reasoning parts", () => {
  const messages: Message[] = [
    {
      id: "1",
      role: "assistant",
      content: "",
      parts: [
        { type: "text", text: "Let me analyze this." },
        {
          type: "reasoning",
          reasoning: "The user wants a button component with specific styling.",
          details: [],
        },
      ],
    },
  ];

  render(<MessageList messages={messages} />);

  expect(screen.getByText("Reasoning")).toBeDefined();
  expect(
    screen.getByText("The user wants a button component with specific styling.")
  ).toBeDefined();
});

test("MessageList renders multiple messages in correct order", () => {
  const messages: Message[] = [
    {
      id: "1",
      role: "user",
      content: "First user message",
    },
    {
      id: "2",
      role: "assistant",
      content: "First assistant response",
    },
    {
      id: "3",
      role: "user",
      content: "Second user message",
    },
    {
      id: "4",
      role: "assistant",
      content: "Second assistant response",
    },
  ];

  const { container } = render(<MessageList messages={messages} />);

  // Get all message containers in order
  const messageContainers = container.querySelectorAll(".rounded-xl");

  // Verify we have 4 messages
  expect(messageContainers).toHaveLength(4);

  // Check the content of each message in order
  expect(messageContainers[0].textContent).toContain("First user message");
  expect(messageContainers[1].textContent).toContain(
    "First assistant response"
  );
  expect(messageContainers[2].textContent).toContain("Second user message");
  expect(messageContainers[3].textContent).toContain(
    "Second assistant response"
  );
});

test("MessageList handles step-start parts", () => {
  const messages: Message[] = [
    {
      id: "1",
      role: "assistant",
      content: "",
      parts: [
        { type: "text", text: "Step 1 content" },
        { type: "step-start" },
        { type: "text", text: "Step 2 content" },
      ],
    },
  ];

  render(<MessageList messages={messages} />);

  expect(screen.getByText("Step 1 content")).toBeDefined();
  expect(screen.getByText("Step 2 content")).toBeDefined();
  // Check that a separator exists (hr element)
  const container = screen.getByText("Step 1 content").closest(".rounded-xl");
  expect(container?.querySelector("hr")).toBeDefined();
});

test("MessageList applies correct styling for user vs assistant messages", () => {
  const messages: Message[] = [
    {
      id: "1",
      role: "user",
      content: "User message",
    },
    {
      id: "2",
      role: "assistant",
      content: "Assistant message",
    },
  ];

  render(<MessageList messages={messages} />);

  const userMessage = screen.getByText("User message").closest(".rounded-xl");
  const assistantMessage = screen
    .getByText("Assistant message")
    .closest(".rounded-xl");

  // User messages should have blue background
  expect(userMessage?.className).toContain("bg-blue-600");
  expect(userMessage?.className).toContain("text-white");

  // Assistant messages should have white background
  expect(assistantMessage?.className).toContain("bg-white");
  expect(assistantMessage?.className).toContain("text-neutral-900");
});

test("MessageList handles empty content with parts", () => {
  const messages: Message[] = [
    {
      id: "1",
      role: "assistant",
      content: "", // Empty content but has parts
      parts: [{ type: "text", text: "This is from parts" }],
    },
  ];

  render(<MessageList messages={messages} />);

  expect(screen.getByText("This is from parts")).toBeDefined();
});

test("MessageList shows loading for assistant message with empty parts", () => {
  const messages: Message[] = [
    {
      id: "1",
      role: "assistant",
      content: "",
      parts: [],
    },
  ];

  const { container } = render(
    <MessageList messages={messages} isLoading={true} />
  );

  // Check that exactly one "Generating..." text appears
  const loadingText = container.querySelectorAll(".text-neutral-500");
  const generatingElements = Array.from(loadingText).filter(
    (el) => el.textContent === "Generating..."
  );
  expect(generatingElements).toHaveLength(1);
});
