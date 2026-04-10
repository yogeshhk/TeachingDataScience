// Note: the "@anthropic-ai/claude-code" package has been renamed
// to "@anthropic-ai/claude-agent-sdk"
import { query } from "@anthropic-ai/claude-agent-sdk";
import path from "path";

const REVIEW_DIR = "src/queries";

async function main() {
  process.exit(0);
  // Read JSON input from stdin
  const input = await new Promise((resolve) => {
    let data = "";
    process.stdin.on("data", (chunk) => (data += chunk));
    process.stdin.on("end", () => resolve(data));
  });

  const hookData = JSON.parse(input);
  const toolInput = hookData.tool_input;

  // Check if this is a file modification in ./queries
  const filePath = toolInput.file_path || toolInput.path;
  if (!filePath) {
    process.exit(0);
  }

  // Normalize paths for comparison
  const normalizedFilePath = path.resolve(filePath);
  const queriesDir = path.resolve(process.cwd(), REVIEW_DIR);

  // Check if file is within queries directory (handles subdirectories too)
  if (!normalizedFilePath.startsWith(queriesDir + path.sep)) {
    process.exit(0);
  }

  // Prepare prompt for analysis
  const newContent =
    toolInput.content || toolInput.contents || toolInput.new_string;
  const prompt = `You are reviewing a proposed change to a database query file.
Your task is to analyze if the new or modified query functions could be 
accomplished by reusing or slightly modifying existing query functions.

Within reason, we want to prevent duplicate queries from being added into this project,
so you are seeing if the proposed change will duplicate any existing functionality.

File: ${filePath}
New content:
<new_content>
${newContent}
</new_content>

Please research and analyze the existing queries in the ./queries directory and:
1. Identify any new query functions being added in this change
2. For each new query function, determine if it could be accomplished by:
   - Using an existing query function as-is
   - Slightly modifying an existing query function, perhaps by adding additional 
      arguments or expanding a select statement

If yes, provide specific feedback on which existing functions could be used instead. Be concise and specific.
If no, just say "Changes look appropriate."`;

  const messages = [];
  for await (const message of query({
    prompt,
  })) {
    messages.push(message);
  }

  // Extract the analysis result
  const resultMessage = messages.find((m) => m.type === "result");
  if (!resultMessage || resultMessage.subtype !== "success") {
    process.exit(0);
  }

  // If changes are appropriate, allow them
  if (resultMessage.result.includes("Changes look appropriate")) {
    process.exit(0);
  }

  // Otherwise, block with feedback
  console.error(`Query duplication detected:\n\n${resultMessage.result}`);
  process.exit(2);
}

main().catch((err) => {
  console.error(`Hook error: ${err.message}`);
  process.exit(1);
});
