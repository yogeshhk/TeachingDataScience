import * as ts from "typescript";
import * as path from "path";

// Read stdin
async function readInput() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString());
}

function runTypeCheck(configPath) {
  // Parse the tsconfig.json file
  const configFile = ts.readConfigFile(configPath, ts.sys.readFile);
  if (configFile.error) {
    console.error(
      ts.formatDiagnostic(configFile.error, {
        getCanonicalFileName: (x) => x,
        getCurrentDirectory: ts.sys.getCurrentDirectory,
        getNewLine: () => ts.sys.newLine,
      })
    );
    return;
  }

  // Parse the configuration
  const parseConfigHost = {
    fileExists: ts.sys.fileExists,
    readFile: ts.sys.readFile,
    readDirectory: ts.sys.readDirectory,
    getCurrentDirectory: ts.sys.getCurrentDirectory,
    onUnRecoverableConfigFileDiagnostic: () => {},
  };

  const parsed = ts.parseJsonConfigFileContent(
    configFile.config,
    parseConfigHost,
    path.dirname(configPath)
  );

  // Override to ensure no emit
  const compilerOptions = {
    ...parsed.options,
    noEmit: true,
  };

  // Create the program
  const program = ts.createProgram(parsed.fileNames, compilerOptions);

  // Get all diagnostics
  const allDiagnostics = ts.getPreEmitDiagnostics(program);

  // Format and display diagnostics
  if (allDiagnostics.length > 0) {
    const formatHost = {
      getCanonicalFileName: (path) => path,
      getCurrentDirectory: ts.sys.getCurrentDirectory,
      getNewLine: () => ts.sys.newLine,
    };

    const formattedDiagnostics = ts.formatDiagnostics(
      allDiagnostics,
      formatHost
    );
    return formattedDiagnostics; // Type check failed
  }

  return null; // Type check passed
}

async function main() {
  const input = await readInput();
  const file = input.tool_response?.filePath || input.tool_input?.file_path;

  // Only check TypeScript files
  if (!file || !/\.(ts|tsx)$/.test(file)) {
    process.exit(0);
  }

  const typeChecks = runTypeCheck("./tsconfig.json");
  if (typeChecks) {
    console.error(typeChecks);
    process.exit(2);
  }
}

main();
