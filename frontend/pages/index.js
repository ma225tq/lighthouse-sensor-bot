import { useState, useEffect, useRef } from "react";
import { useWebSocket } from "../contexts/WebSocketContext";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import ReactMarkdown from 'react-markdown';
import { marked } from 'marked';

export default function QuestionForm() {
  const [question, setQuestion] = useState("");
  const [content, setContent] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState("offline");
  const [sourceFile, setSourceFile] = useState("ferry_trips_data.csv");
  const [modelCategory, setModelCategory] = useState("open-source");
  const [selectedModel, setSelectedModel] = useState("qwen/qwen-2.5-72b-instruct");
  const [modelUsed, setModelUsed] = useState(null);
  const [activeQuery, setActiveQuery] = useState(false);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [fullResponse, setFullResponse] = useState(null);
  const { sqlQueries, queryStatus, resetQueries, evaluationProgress } = useWebSocket();
  const [controlMode, setControlMode] = useState("query"); // "query" or "evaluation"
  const [testCases, setTestCases] = useState(null);

  const [queryModeState, setQueryModeState] = useState({
    content: null,
    fullResponse: null,
    sqlQueries: [],
    modelUsed: null,
    activeQuery: false
  });

  const [evaluationModeState, setEvaluationModeState] = useState({
    content: null,
    fullResponse: null,
    evaluationResults: null,
    sqlQueries: [],
    activeQuery: false
  });

  const fetchTestCases = async () => {
    try {
      const response = await fetch("/api/test-cases");
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      const data = await response.json();
      setTestCases(data);
    } catch (error) {
      console.error("Error fetching test cases:", error);
      setTestCases({ error: error.message });
    }
  };

  // Fetch test cases when switching to evaluation mode
  useEffect(() => {
    if (controlMode === "evaluation" && !testCases) {
      fetchTestCases();
    }
  }, [controlMode, testCases]);


  const markdownToHtml = (markdown) => {
    if (!markdown) return '';
    try {
      return marked(markdown);
    } catch (error) {
      console.error('Error parsing markdown:', error);
      return markdown;
    }
  };

  // Reference to the SQL queries container for auto-scrolling
  const queriesContainerRef = useRef(null);

  // Auto-scroll when new queries are added
  useEffect(() => {
    if (queriesContainerRef.current) {
      queriesContainerRef.current.scrollTop = queriesContainerRef.current.scrollHeight;
    }
  }, [sqlQueries]);
  useEffect(() => {
    // Check backend status on component mount
    testConnection(false);
  }, []);

  useEffect(() => {
    // Set Live Tool Calls as the default tab when component mounts
    const timer = setTimeout(() => {
      const toolCallsTab = document.querySelector('.tab-button[data-tab="live-tool-calls"]');
      if (toolCallsTab) {
        toolCallsTab.classList.add('active');
      }

      const toolCallsContent = document.getElementById('live-tool-calls-content');
      if (toolCallsContent) {
        toolCallsContent.classList.remove('hidden');
        toolCallsContent.classList.add('active');
      }

      const evalContent = document.getElementById('evaluation-content');
      if (evalContent) {
        evalContent.classList.add('hidden');
        evalContent.classList.remove('active');
      }
    }, 100);

    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    // Save current state when switching modes
    if (controlMode === "query") {
      // Coming from evaluation mode, save evaluation state
      setEvaluationModeState({
        content,
        fullResponse,
        evaluationResults,
        sqlQueries,
        activeQuery
      });

      // Restore query mode state
      setContent(queryModeState.content);
      setFullResponse(queryModeState.fullResponse);
      setModelUsed(queryModeState.modelUsed);
      setActiveQuery(queryModeState.activeQuery);

      // Reset evaluation-specific state
      setEvaluationResults(null);

      // Reset SQL queries display to show query mode queries
      resetQueries(); // Clear the current display

      // Switch to full response tab when switching to query mode
      // Only if there is a full response, otherwise show live tool calls
      Promise.resolve().then(() => {
        if (queryModeState.fullResponse) {
          switchTab('full-response');
        } else {
          switchTab('live-tool-calls');
        }
      });
    } else {
      // Coming from query mode, save query state
      setQueryModeState({
        content,
        fullResponse,
        modelUsed,
        sqlQueries,
        activeQuery
      });

      // Restore evaluation mode state
      setContent(evaluationModeState.content);
      setFullResponse(evaluationModeState.fullResponse);
      setEvaluationResults(evaluationModeState.evaluationResults);
      setActiveQuery(evaluationModeState.activeQuery);

      // Reset SQL queries display to show evaluation mode queries
      resetQueries(); // Clear the current display

      // Switch to evaluation tab when switching to evaluation mode
      // Only if evaluation results exist, otherwise show full response or live tool calls
      Promise.resolve().then(() => {
        if (evaluationModeState.evaluationResults) {
          switchTab('evaluation');
        } else if (evaluationModeState.fullResponse) {
          switchTab('full-response');
        } else {
          switchTab('live-tool-calls');
        }
      });
    }
  }, [controlMode]);

  const askQuestion = async () => {
    if (question.trim() === "") {
      setContent("Please enter a question");
      return;
    }

    setIsLoading(true);
    setContent(null);
    setActiveQuery(true);
    resetQueries(); // Reset SQL queries for new question

    // Switch to Live Tool Calls tab to show progress
    switchTab('live-tool-calls');

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          source_file: sourceFile,
          llm_model_id: selectedModel
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();

      // First update all state values
      setContent(data.content);
      setFullResponse(data.full_response);
      setModelUsed(selectedModel);

      // Update query mode state with all relevant data
      setQueryModeState({
        content: data.content,
        fullResponse: data.full_response,
        modelUsed: selectedModel,
        sqlQueries: sqlQueries,
        activeQuery: true
      });

      // Use React's state batching to our advantage
      // This will run after the state updates are applied
      Promise.resolve().then(() => {
        switchTab('full-response');
      });
    } catch (error) {
      console.error("Error asking question:", error);
      setContent("Error connecting to the backend: " + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const testConnection = async (showAlert = true) => {
    try {
      const response = await fetch("/api/test");
      const data = await response.json();

      if (data.status === "online" || data.content === "OK" ||
        data.content === "Backend connection test successful") {
        setBackendStatus("online");
        if (showAlert) {
          alert(`Backend connection test: Success`);
        }
      } else {
        setBackendStatus("offline");
        if (showAlert) {
          alert(`Backend connection test: Failed - ${data.content || "Unknown error"}`);
        }
      }
    } catch (error) {
      setBackendStatus("offline");
      if (showAlert) {
        alert(`Backend connection test failed: ${error.message}`);
      }
    }
  };

  const loadPrompt = async () => {
    try {
      const response = await fetch("/api/load-prompt");
      const data = await response.json();

      if (data.content) {
        setQuestion(data.content);
      } else {
        alert("Failed to load prompt: No content");
      }
    } catch (error) {
      alert(`Failed to load prompt: ${error.message}`);
    }
  };

  const handleCategoryChange = (category) => {
    setModelCategory(category);
    // Set a default model for the selected category
    if (category === "proprietary") {
      setSelectedModel("openai/gpt-4o-2024-11-20");
    } else {
      setSelectedModel("qwen/qwen-2.5-72b-instruct");
    }
  };

  // Update the backend status indicator with breathing effect
  const statusIndicator = (
    <div className="flex items-center">
      <span
        className={`backend-status-indicator ${backendStatus === "online" ? "online" : "offline"}`}
      ></span>
      <span className="ml-2 text-sm text-white text-opacity-80">
        {backendStatus === "online" ? "Backend connected" : "Backend disconnected"}
      </span>
    </div>
  );
  // Add this function to handle tab switching
  const switchTab = (tabName) => {
    // Remove active class from all tabs
    document.querySelectorAll('.tab-button').forEach(tab => {
      tab.classList.remove('active');
    });

    // Find the button for this tab and make it active
    const tabButton = document.querySelector(`.tab-button[data-tab="${tabName}"]`);
    if (tabButton) {
      tabButton.classList.add('active');
    }

    // Hide all tab content
    document.querySelectorAll('.tab-pane').forEach(content => {
      content.classList.remove('active');
      content.classList.add('hidden');
    });

    // Find the content for this tab and make it visible
    const tabContent = document.getElementById(`${tabName}-content`);
    if (tabContent) {
      tabContent.classList.remove('hidden');
      tabContent.classList.add('active');
    } else {
      console.warn(`Tab content for "${tabName}" not found`);
    }
  };

  // section to display SQL queries with syntax highlighting
  const renderSqlQueries = () => {
    if (sqlQueries.length === 0) return null;

    return (
      <div className="mt-4 bg-white bg-opacity-10 rounded-xl p-4">
        <div
          ref={queriesContainerRef}
          className="space-y-2 max-h-60 overflow-y-auto"
        >
          {sqlQueries.map((query, index) => (
            <div key={index} className="rounded">
              <SyntaxHighlighter
                language="sql"
                style={vscDarkPlus}
                customStyle={{
                  margin: 0,
                  borderRadius: '0.25rem',
                  fontSize: '0.875rem',
                  padding: '0.5rem'
                }}
              >
                {query}
              </SyntaxHighlighter>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const evaluateModel = async () => {
    if (!selectedModel) {
      alert("Please select a model to evaluate");
      return;
    }

    setIsLoading(true);

    switchTab('live-tool-calls');

    try {
      setContent("Running model evaluation...");

      // Reset evaluation progress
      resetQueries();

      const response = await fetch("/api/evaluate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model_id: selectedModel
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        // Only update evaluation mode state, not query mode state
        setContent(`## Error during evaluation\n${data.error}`);
        setFullResponse(`## Error during evaluation\n${data.error}`);
        setEvaluationResults(null);

        setEvaluationModeState({
          content: `## Error during evaluation\n${data.error}`,
          fullResponse: `## Error during evaluation\n${data.error}`,
          evaluationResults: null,
          sqlQueries: sqlQueries,
          activeQuery: true
        });
      } else {
        // Only update evaluation mode state, not query mode state
        setEvaluationResults(data.results);
        setContent("## Evaluation successful! ✓\n\nDetailed results available in the Evaluation tab.");

        if (data.full_response) {
          setFullResponse(data.full_response);
        }

        setEvaluationModeState({
          content: "## Evaluation successful! ✓\n\nDetailed results available in the Evaluation tab.",
          fullResponse: data.full_response || null,
          evaluationResults: data.results,
          sqlQueries: sqlQueries,
          activeQuery: true
        });

        // Switch to evaluation tab after a short delay to ensure state updates are processed
        setTimeout(() => {
          switchTab('evaluation');
          // Force the evaluation content to be visible
          const evaluationContent = document.getElementById('evaluation-content');
          if (evaluationContent) {
            document.querySelectorAll('.tab-pane').forEach(content => {
              content.classList.remove('active');
              content.classList.add('hidden');
            });

            evaluationContent.classList.remove('hidden');
            evaluationContent.classList.add('active');

            // Force a re-render of the evaluation container
            const evaluationContainer = document.getElementById('evaluation-data-container');
            if (evaluationContainer) {
              const displayStyle = evaluationContainer.style.display;
              evaluationContainer.style.display = 'none';
              setTimeout(() => {
                evaluationContainer.style.display = displayStyle || 'block';
              }, 10);
            }
          }
        }, 100);
      }
    } catch (error) {
      console.error("Error evaluating model:", error);
      setContent(`## Error during evaluation\n${error.message}`);
      setFullResponse(`## Error during evaluation\n${error.message}`);
      setEvaluationResults(null);
    } finally {
      setIsLoading(false);
    }
  };

  // Add new function to run 10 iterations
  const run10Iterations = async () => {
    if (!selectedModel) {
      alert("Please select a model to evaluate");
      return;
    }

    setIsLoading(true);
    switchTab('live-tool-calls');
    setContent("Running 10 model evaluation iterations...");
    resetQueries();

    try {
      for (let i = 0; i < 10; i++) {
        setContent(`Running iteration ${i+1} of 10...`);
        
        const response = await fetch("/api/evaluate", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model_id: selectedModel
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status} in iteration ${i+1}`);
        }

        await response.json(); // We don't need to process each response, just wait for completion
        
        // Update progress without using socketio directly
        // The backend will emit its own progress updates through the WebSocket connection
        const progressPercent = ((i + 1) / 10) * 100;
        // Just update the content to show progress
        setContent(`## Iteration ${i+1} of 10 completed (${progressPercent.toFixed(0)}%)\n\nRunning next iteration...`);
      }
      
      setContent("## All 10 evaluation iterations completed! ✓\n\nCheck Model Performance dashboard to see aggregated results.");
      
    } catch (error) {
      console.error("Error during 10 iterations:", error);
      setContent(`## Error during 10 iterations\n${error.message}`);
      setFullResponse(`## Error during 10 iterations\n${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  function EvaluationResultsTable({ results }) {
    if (!results) {
      // console.log("No evaluation results provided");
      return null;
    }

    // console.log("Rendering evaluation results:", results);


    const processMetrics = (data) => {

      const metricsData = data.results ? data.results : data;

      return Object.entries(metricsData)
        .filter(([key, value]) =>
          (typeof value === 'number' || typeof value === 'boolean') &&
          !['id', 'query_result_id', 'retrieved_contexts', 'reference', 'hash'].includes(key)
        )
        .map(([key, value]) => ({
          key: key,
          value
        }));
    };

    const metricsData = processMetrics(results);

    return (
      <div className="mt-4 overflow-hidden rounded-lg border border-gray-200 shadow">
        <table className="w-full border-collapse bg-white text-left">
          <thead className="bg-gray-50">
            <tr>
              <th scope="col" className="px-4 py-3 text-sm font-medium text-gray-900">Metric</th>
              <th scope="col" className="px-4 py-3 text-sm font-medium text-gray-900">Value</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 border-t border-gray-100">
            {metricsData.length > 0 ? (
              metricsData.map((item) => (
                <tr key={item.key} className="hover:bg-gray-50">
                  <td className="px-4 py-2 text-sm font-medium text-gray-700">
                    {formatMetricName(item.key)}
                  </td>
                  <td className="px-4 py-2 text-sm text-gray-700">
                    {typeof item.value === 'number' ? item.value.toFixed(2) : String(item.value)}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="2" className="px-4 py-4 text-sm text-center text-gray-500">
                  No metrics available
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    );
  }

  // Helper function to format metric names
  const formatMetricName = (key) => {
    return key
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Add this function to render the progress bar
  const renderEvaluationProgress = () => {
    if (!evaluationProgress) return null;

    const { progress, total, percent, message } = evaluationProgress;
    const displayPercent = percent !== undefined ? percent : Math.round((progress / total) * 100);

    return (
      <div className="mt-4 mb-4">
        <div className="flex justify-between mb-1">
          <span className="text-sm font-medium text-blue-700">{message} - {displayPercent}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div
            className="bg-blue-600 h-2.5 rounded-full transition-all duration-300 ease-in-out"
            style={{ width: `${displayPercent}%` }}
          ></div>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-ferry-image min-h-screen">
      <main className="container mx-auto py-6 flex justify-center">
        <div className="grid grid-cols-[30vw_60vw] gap-5">
          <div>
            <div className="sidebar-container rounded-xl p-6 bg-white bg-opacity-95 shadow-lg border border-gray-100">
              <div className="flex items-center mb-6">
                <div className="p-2 bg-blue-600 rounded text-white mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17l4-4m0 0l4-4m-4 4H3m4 4h10" />
                  </svg>
                </div>
                <h1 className="text-xl font-bold text-gray-800">Query Controls</h1>
              </div>

              <div className="mb-4">
                <div className="flex items-center mb-2">
                  <div className="flex rounded-lg overflow-hidden border border-gray-200 flex-grow">
                    <button
                      className={`flex-1 py-2 px-4 text-center transition-colors ${controlMode === "query"
                          ? "bg-blue-600 text-white"
                          : isLoading
                            ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                            : "bg-gray-100 text-gray-600"
                        }`}
                      onClick={() => !isLoading && setControlMode("query")}
                      disabled={isLoading}
                    >
                      Query Mode
                    </button>
                    <button
                      className={`flex-1 py-2 px-4 text-center transition-colors ${controlMode === "evaluation"
                          ? "bg-blue-600 text-white"
                          : isLoading
                            ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                            : "bg-gray-100 text-gray-600"
                        }`}
                      onClick={() => !isLoading && setControlMode("evaluation")}
                      disabled={isLoading}
                    >
                      Evaluation Mode
                    </button>
                  </div>
                  <div className="relative ml-2 group">
                    <div className="w-6 h-6 bg-gray-200 rounded-full flex items-center justify-center cursor-help">
                      <span className="text-gray-600 font-semibold">?</span>
                    </div>
                    <div className="absolute left-full ml-2 top-1/2 transform -translate-y-1/2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-10 pointer-events-none">
                      <p className="mb-1"><strong>Query Mode:</strong> Ask questions about the ferry data and get AI-generated analysis.</p>
                      <p><strong>Evaluation Mode:</strong> Test the model's performance against predefined test cases to measure accuracy and reliability.</p>
                      {isLoading && <p className="mt-1 text-yellow-300">Mode switching is disabled while a query is running.</p>}
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-5">
                <div>

                  <div className="mb-2 mt-5">

                    <div className="flex rounded-lg overflow-hidden border border-gray-200">
                      <button
                        className={`flex-1 py-2 px-4 text-center transition-colors ${modelCategory === "proprietary" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-600"}`}
                        onClick={() => handleCategoryChange("proprietary")}
                      >
                        Proprietary
                      </button>
                      <button
                        className={`flex-1 py-2 px-4 text-center transition-colors ${modelCategory === "open-source" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-600"}`}
                        onClick={() => handleCategoryChange("open-source")}
                      >
                        Open Source
                      </button>
                    </div>
                  </div>

                  <div className="mb-4">
                    <label className="text-sm font-medium text-gray-700 block mb-2">Model</label>
                    <div className="relative">
                      <select
                        className="block w-full rounded-lg border border-gray-300 px-4 py-2.5 text-gray-700 appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                      >
                        {modelCategory === "proprietary" ? (
                          <>
                            <option value="openai/gpt-4o-2024-11-20">OpenAI GPT-4o</option>
                            <option value="anthropic/claude-3.7-sonnet">Claude 3.7 Sonnet</option>
                            <option value="google/gemini-2.0-flash-001">Gemini 2.0 Flash</option>
                            <option value="amazon/nova-pro-v1">Amazon Nova Pro</option>
                          </>
                        ) : (
                          <>
                            <option value="qwen/qwen-2.5-72b-instruct">Qwen 2.5 72B</option>
                            <option value="meta-llama/llama-3.3-70b-instruct">Llama 3.3 70B</option>
                            <option value="meta-llama/llama-3.1-8b-instruct">Llama 3.1 8B</option>
                            <option value="mistralai/ministral-8b">Mistral 8B</option>
                          </>
                        )}
                      </select>
                      <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                        </svg>
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium text-gray-700 block mb-2">Data Source</label>
                  <div className="block w-full rounded-lg border border-gray-300 px-4 py-2.5 text-gray-700 bg-gray-50">
                    Ferry Trips Data (CSV)
                  </div>
                </div>

                {controlMode === "query" ? (
                  <>
                    <div>
                      <label className="text-sm font-medium text-gray-700 block mb-2">Your Analysis Query</label>
                      <textarea
                        className="w-full h-32 px-3 py-2 text-gray-700 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="E.g., What is the average speed of ferry Jupiter? How does fuel consumption correlate with passenger load?"
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                      ></textarea>
                    </div>

                    <div className="flex space-x-3 pt-2">
                      <div className="relative group">
                        <button
                          className="flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                          onClick={loadPrompt}
                          aria-label="Load an example query into the input field"
                        >
                          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                          </svg>
                          Example
                        </button>
                        <div className="absolute bottom-full left-0 mb-2 w-48 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-10">
                          Click to load a pre-written example query into the input field
                        </div>
                      </div>

                      <button
                        className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                        onClick={() => setQuestion("")}
                      >
                        Clear
                      </button>

                      <button
                        className="flex-1 flex items-center justify-center px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
                        onClick={askQuestion}
                        disabled={isLoading || !question.trim()}
                      >
                        <span>Query</span>
                        <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path>
                        </svg>
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <div>
                      <label className="text-sm font-medium text-gray-700 block mb-2">Test Cases</label>
                      <div className="w-full h-96 border rounded-lg overflow-y-auto bg-gray-50">
                        {testCases ? (
                          testCases.error ? (
                            <p className="text-red-500 p-3">Error loading test cases: {testCases.error}</p>
                          ) : (
                            <SyntaxHighlighter
                              language="json"
                              style={vscDarkPlus}
                              customStyle={{
                                margin: 0,
                                borderRadius: '0.25rem',
                                fontSize: '0.875rem',
                                padding: '0.5rem',
                                height: '100%'
                              }}
                            >
                              {JSON.stringify(testCases, null, 2)}
                            </SyntaxHighlighter>
                          )
                        ) : (
                          <div className="flex justify-center items-center h-full">
                            <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-blue-500"></div>
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="flex flex-col space-y-3 mt-3">
                      <button
                        className="w-full flex items-center justify-center px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
                        onClick={evaluateModel}
                        disabled={isLoading}
                      >
                        <span>Evaluate Model (Single Run)</span>
                        <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                        </svg>
                      </button>
                      
                      <button
                        className="w-full flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                        onClick={run10Iterations}
                        disabled={isLoading}
                      >
                        <span>Kör 10 iterationer</span>
                        <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                        </svg>
                      </button>
                    </div>
                  </>
                )}

  
              </div>
            </div>
          </div>

          <div>
            <div className="transparent-card rounded-xl p-5 shadow-xl border border-gray-600 border-opacity-30 h-full">
              <div className="flex items-center mb-5">
                <div className="p-2 bg-blue-600 rounded text-white mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" stroke="white" />
                    <polyline points="22 4 12 14.01 9 11.01" stroke="white" fill="none" />
                  </svg>
                </div>
                <h2 className="text-xl font-bold">Response & Analysis Results</h2>
              </div>

              <div className="mb-4 visualization-container">
                <div className="response-container rounded-lg p-3 max-h-48 overflow-y-auto">
                  {isLoading ? (
                    <div className="flex flex-col justify-center items-center py-4">
                      {evaluationProgress ? (
                        renderEvaluationProgress()
                      ) : (
                        <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-black mb-3"></div>
                      )}
                    </div>
                  ) : content ? (
                    <div className="prose prose-sm max-w-none text-gray-800">
                      <ReactMarkdown>{content}</ReactMarkdown>
                    </div>
                  ) : (
                    <div className="text-gray-500 text-sm text-center py-4">
                      Response will appear here
                    </div>
                  )}
                </div>
              </div>

              <div className="mb-4 border-b border-white border-opacity-20">
                <div className="flex space-x-2 border-b border-white border-opacity-20">
                  <button
                    className="tab-button px-4 py-2"
                    data-tab="live-tool-calls"
                    onClick={() => switchTab('live-tool-calls')}
                  >
                    Live Tool Calls
                  </button>
                  <button
                    className="tab-button px-4 py-2"
                    data-tab="full-response"
                    onClick={() => switchTab('full-response')}
                  >
                    Full Response
                  </button>
                  {controlMode === "evaluation" && (
                    <button
                      className="tab-button px-4 py-2"
                      data-tab="evaluation"
                      onClick={() => switchTab('evaluation')}
                    >
                      Evaluation
                    </button>
                  )}
                </div>
              </div>

              <div
                className="bg-white bg-opacity-20 rounded-xl p-6 visualization-container visualization-expanded overflow-x-auto"
              >
                {isLoading ? (
                  <div id="tab-content" className="h-full">
                    <div id="live-tool-calls-content" className="tab-pane active">
                      <div className="h-full w-full flex flex-col">
                        <div className="flex justify-center items-center mb-4">
                          <div className="animate-spin rounded-full h-8 w-8 border-4 border-white border-opacity-20 border-t-white mr-3"></div>
                        </div>
                        <div id="sql-data-container" className="w-full h-full flex-1">
                          {sqlQueries.length > 0 ? (
                            <div className="bg-white bg-opacity-10 rounded-xl p-4 h-full w-full">
                              <div
                                ref={queriesContainerRef}
                                className="space-y-2 overflow-y-auto h-[calc(100%-2rem)] w-full"
                              >
                                {sqlQueries.map((query, index) => (
                                  <div key={index} className="rounded w-full">
                                    <SyntaxHighlighter
                                      language="sql"
                                      style={vscDarkPlus}
                                      customStyle={{
                                        margin: 0,
                                        borderRadius: '0.25rem',
                                        fontSize: '0.875rem',
                                        padding: '0.5rem',
                                        width: '100%',
                                        maxWidth: '100%'
                                      }}
                                    >
                                      {query}
                                    </SyntaxHighlighter>
                                  </div>
                                ))}
                              </div>
                            </div>
                          ) : (
                            <div className="flex items-center justify-center h-full">
                              <p className="text-gray-300 text-center">No tool calls executed yet</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : content ? (
                  <div id="tab-content" className="h-full">
                    <div id="live-tool-calls-content" className="tab-pane active">
                      <div className="h-full w-full flex flex-col">
                        <div id="sql-data-container" className="w-full h-full">
                          {sqlQueries.length > 0 ? (
                            <div className="bg-white bg-opacity-10 rounded-xl p-4 h-full">
                              <div
                                ref={queriesContainerRef}
                                className="space-y-2 overflow-y-auto h-[calc(100%-2rem)]"
                              >
                                {sqlQueries.map((query, index) => (
                                  <div key={index} className="rounded">
                                    <SyntaxHighlighter
                                      language="sql"
                                      style={vscDarkPlus}
                                      customStyle={{
                                        margin: 0,
                                        borderRadius: '0.25rem',
                                        fontSize: '0.875rem',
                                        padding: '0.5rem'
                                      }}
                                    >
                                      {query}
                                    </SyntaxHighlighter>
                                  </div>
                                ))}
                              </div>
                            </div>
                          ) : (
                            <div className="flex items-center justify-center h-full">
                              <p className="text-gray-300 text-center">No tool calls executed yet</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    <div id="full-response-content" className="tab-pane hidden">
                      <div className="h-full w-full flex flex-col">
                        <div className="w-full h-full overflow-y-auto bg-white bg-opacity-10 rounded-xl p-4">
                          <ReactMarkdown
                            className="prose prose-sm max-w-none text-gray-200"
                            components={{
                              code({ node, inline, className, children, ...props }) {
                                const match = /language-(\w+)/.exec(className || '')
                                const language = match ? match[1] : 'text'
                                return !inline && (language === 'sql' || language === 'mysql') ? (
                                  <SyntaxHighlighter
                                    language="sql"
                                    style={vscDarkPlus}
                                    customStyle={{
                                      margin: 0,
                                      borderRadius: '0.25rem',
                                      fontSize: '0.875rem',
                                      padding: '0.5rem'
                                    }}
                                    {...props}
                                  >
                                    {String(children).replace(/\n$/, '')}
                                  </SyntaxHighlighter>
                                ) : (
                                  <code className="bg-gray-200 px-1 py-0.5 rounded text-gray-800" {...props}>
                                    {children}
                                  </code>
                                )
                              }
                            }}
                          >
                            {fullResponse || 'No response available'}
                          </ReactMarkdown>
                        </div>
                      </div>
                    </div>

                    <div id="evaluation-content" className="tab-pane hidden">
                      <div className="h-full w-full flex items-center justify-center">
                        <div id="evaluation-data-container" className="w-full h-full">
                          {evaluationResults ? (
                            <div className="p-4">
                              <div className="mb-4 text-center">
                                <h3 className="text-lg font-semibold">Evaluation Results for {selectedModel}</h3>
                                <p className="text-green-600 font-medium">Evaluation successful!</p>
                              </div>
                              <EvaluationResultsTable results={evaluationResults} />

                              {/* If there are retrieved contexts, display them */}
                              {evaluationResults.retrieved_contexts && (
                                <div className="mt-6">
                                  <h4 className="text-md font-semibold mb-2">Retrieved Contexts</h4>
                                  <div className="bg-gray-50 p-3 rounded text-sm">
                                    {evaluationResults.retrieved_contexts}
                                  </div>
                                </div>
                              )}
                            </div>
                          ) : (
                            <p className="text-gray-500 text-center">Something went wrong</p>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col justify-center items-center h-full">
                    <svg className="w-24 h-24 text-white text-opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M8 13v-1m4 1v-3m4 3V8M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"></path>
                    </svg>
                    <h3 className="mt-4 text-white text-opacity-80 text-lg font-medium">Run an analysis to see results</h3>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}