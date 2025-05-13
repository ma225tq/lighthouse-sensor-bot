import React, { useState, useEffect, forwardRef, useImperativeHandle } from 'react';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend, 
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler
} from 'chart.js';
import { Bar, Radar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale, 
  LinearScale, 
  BarElement, 
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Title, 
  Tooltip, 
  Legend
);

const COLORS = [
  'rgba(31, 119, 180, 0.8)',
  'rgba(255, 127, 14, 0.8)',
  'rgba(44, 160, 44, 0.8)',
  'rgba(214, 39, 40, 0.8)',
  'rgba(148, 103, 189, 0.8)',
  'rgba(140, 86, 75, 0.8)',
  'rgba(227, 119, 194, 0.8)',
  'rgba(127, 127, 127, 0.8)',
  'rgba(188, 189, 34, 0.8)',
  'rgba(23, 190, 207, 0.8)'
];

const ModelPerformanceChart = forwardRef(function ModelPerformanceChart(props, ref) {
  const { setPageLoading } = props;
  const [performanceData, setPerformanceData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState('avg_factual_correctness');
  const [chartType, setChartType] = useState('bar');
  const [modelTypeFilter, setModelTypeFilter] = useState(null);
  const [metricType, setMetricType] = useState('performance');
  const [isExporting, setIsExporting] = useState(false);
  
  const fetchData = async () => {
    try {
      setLoading(true);
      if (setPageLoading) setPageLoading(true);
      
      const params = new URLSearchParams();
      if (modelTypeFilter) {
        params.append('type', modelTypeFilter);
      }
      
      const response = await fetch(`/api/model-performance?${params.toString()}`);
      
      if (!response.ok) {
        throw new Error(`Error fetching model performance: ${response.statusText}`);
      }
      
      const data = await response.json();
      setPerformanceData(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching model performance:', err);
      setError('Failed to load model performance data');
    } finally {
      setLoading(false);
      if (setPageLoading) setPageLoading(false);
    }
  };
  
  useImperativeHandle(ref, () => ({
    refreshData: fetchData
  }));
  
  useEffect(() => {
    fetchData();
  }, [modelTypeFilter]);

  const handleMetricChange = (e) => {
    setSelectedMetric(e.target.value);
  };

  const handleChartTypeChange = (e) => {
    setChartType(e.target.value);
  };

  const handleModelTypeChange = (e) => {
    setModelTypeFilter(e.target.value === 'all' ? null : e.target.value);
  };

  const handleMetricTypeChange = (e) => {
    const newMetricType = e.target.value;
    setMetricType(newMetricType);
    
    if (newMetricType === 'token') {
      // Default to avg_total_tokens when switching to token metrics
      setSelectedMetric('avg_total_tokens');
    } else {
      // Default to first performance metric when switching to performance metrics
      const perfMetrics = performanceData.metrics.filter(m => 
        !['avg_prompt_tokens', 'avg_completion_tokens', 'avg_total_tokens'].includes(m.id) && 
        m.id.startsWith('avg_'));
      if (perfMetrics.length > 0) {
        setSelectedMetric(perfMetrics[0].id);
      }
    }
  };
  
  const handleExportChart = async () => {
    try {
      setIsExporting(true);
      
      // Prepare data for export
      const requestData = {
        chartType,
        metricId: selectedMetric,
        modelType: modelTypeFilter,
        models: performanceData.data
      };
      
      // Call the export endpoint
      const response = await fetch('/api/export-chart', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Error exporting chart: ${errorData.error || response.statusText}`);
      }
      
      const data = await response.json();
      
      // Create download link
      const downloadLink = document.createElement('a');
      downloadLink.href = `data:image/${data.format};base64,${data.image}`;
      
      // Generate filename with current date and chart details
      const metricName = performanceData.metrics.find(m => m.id === selectedMetric)?.name || selectedMetric;
      const date = new Date().toISOString().split('T')[0];
      downloadLink.download = `${date}_llm_performance_${metricName.toLowerCase().replace(/\s+/g, '_')}.${data.format}`;
      
      // Trigger download
      document.body.appendChild(downloadLink);
      downloadLink.click();
      document.body.removeChild(downloadLink);
      
    } catch (err) {
      console.error('Error exporting chart:', err);
      alert(`Failed to export chart: ${err.message}`);
    } finally {
      setIsExporting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded">
        <p>{error}</p>
      </div>
    );
  }

  if (!performanceData || !performanceData.data || performanceData.data.length === 0) {
    return (
      <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 rounded">
        <p>No performance data available.</p>
      </div>
    );
  }

  // Get the corresponding stddev field for a metric
  const getStdDevField = (metricField) => {
    return metricField.replace('avg_', 'stddev_');
  };

  // Prepare bar chart data
  const prepareBarChartData = () => {
    if (!performanceData) return { labels: [], datasets: [] };

    const isTokenMetric = ['avg_prompt_tokens', 'avg_completion_tokens', 'avg_total_tokens'].includes(selectedMetric);
    
    return {
      labels: performanceData.data.map(model => model.model_name.split('/')[1]),
      datasets: [
        {
          label: performanceData.metrics.find(m => m.id === selectedMetric)?.name || selectedMetric,
          data: performanceData.data.map(model => model[selectedMetric]),
          backgroundColor: performanceData.data.map((_, idx) => COLORS[idx % COLORS.length]),
          borderWidth: 1,
        }
      ]
    };
  };

  const prepareRadarChartData = () => {
    if (!performanceData) return { labels: [], datasets: [] };

    const performanceMetrics = performanceData.metrics.filter(m => 
      m.id.startsWith('avg_') && 
      !['avg_prompt_tokens', 'avg_completion_tokens', 'avg_total_tokens'].includes(m.id)
    );

    return {
      labels: performanceMetrics.map(metric => metric.name),
      datasets: performanceData.data.map((model, idx) => ({
        label: model.model_name.split('/')[1],
        data: performanceMetrics.map(metric => model[metric.id]),
        backgroundColor: `${COLORS[idx % COLORS.length].replace('0.8', '0.2')}`,
        borderColor: COLORS[idx % COLORS.length],
        borderWidth: 2,
      }))
    };
  };

  const formatStdDev = (value) => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'string') {
      const parsed = parseFloat(value);
      return isNaN(parsed) ? 'N/A' : parsed.toFixed(3);
    }
    return typeof value === 'number' ? value.toFixed(3) : 'N/A';
  };

  const isTokenMetric = ['avg_prompt_tokens', 'avg_completion_tokens', 'avg_total_tokens'].includes(selectedMetric);

  return (
    <div className="transparent-card rounded-xl p-5 shadow-xl border border-gray-600 border-opacity-30 w-full">
      <div className="mb-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Chart Type
            </label>
            <select
              value={chartType}
              onChange={handleChartTypeChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="bar">Bar Chart</option>
              <option value="radar">Radar Chart</option>
            </select>
          </div>
          
          {
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Metric Type
                </label>
                <select
                  value={metricType}
                  onChange={handleMetricTypeChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="performance">Performance Metrics</option>
                  <option value="token">Token Usage</option>
                </select>
              </div>
              
              {metricType === 'token' ? (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Token Metric
                  </label>
                  <select
                    value={selectedMetric}
                    onChange={handleMetricChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="avg_prompt_tokens">Prompt Tokens</option>
                    <option value="avg_completion_tokens">Completion Tokens</option>
                    <option value="avg_total_tokens">Total Tokens</option>
                  </select>
                </div>
              ) : (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Performance Metric
                  </label>
                  <select
                    value={selectedMetric}
                    onChange={handleMetricChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  >
                    {performanceData.metrics.filter(m => 
                      m.id.startsWith('avg_') && 
                      !['avg_prompt_tokens', 'avg_completion_tokens', 'avg_total_tokens'].includes(m.id)
                    ).map(metric => (
                      <option key={metric.id} value={metric.id}>
                        {metric.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Model Type
                </label>
                <select
                  value={modelTypeFilter || 'all'}
                  onChange={handleModelTypeChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="all">All Models</option>
                  <option value="openai">OpenAI</option>
                  <option value="deepseek">DeepSeek</option>
                </select>
              </div>
            </>
          }
        </div>
        
        {/* Download Button */}
        <div className="flex justify-end mb-4">
          <button
            onClick={handleExportChart}
            disabled={isExporting}
            className="flex items-center bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md transition duration-200"
          >
            {isExporting ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </>
            ) : (
              <>
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                </svg>
                Download High-Quality Chart
              </>
            )}
          </button>
        </div>
      </div>
      
      <div className="h-96">
        {chartType === 'bar' ? (
          <Bar 
            data={prepareBarChartData()} 
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  position: 'top',
                },
                title: {
                  display: true,
                  text: `Model ${isTokenMetric ? 'Token Usage' : 'Performance'}: ${performanceData.metrics.find(m => m.id === selectedMetric)?.name || selectedMetric}`,
                  font: {
                    size: 16
                  }
                },
                tooltip: {
                  callbacks: {
                    label: function(context) {
                      const model = performanceData.data[context.dataIndex];
                      if (isTokenMetric) {
                        const value = context.parsed.y;
                        return `${context.dataset.label}: ${value ? Math.round(value).toLocaleString() : 'N/A'}`;
                      }
                      return `${context.dataset.label}: ${context.parsed.y.toFixed(3)}`;
                    },
                    afterLabel: function(context) {
                      const model = performanceData.data[context.dataIndex];
                      const stdDevField = getStdDevField(selectedMetric);
                      const stdDev = model[stdDevField];
                      if (isTokenMetric) {
                        const value = parseFloat(stdDev);
                        return `Standard Deviation: ${!isNaN(value) ? Math.round(value).toLocaleString() : 'N/A'}`;
                      }
                      return `Standard Deviation: ${formatStdDev(stdDev)}`;
                    }
                  }
                }
              },
              scales: {
                y: {
                  beginAtZero: true,
                  max: isTokenMetric ? undefined : 1,
                  title: {
                    display: true,
                    text: isTokenMetric ? 'Average Tokens' : 'Score (0-1)'
                  }
                }
              }
            }}
          />
        ) : (
          <Radar
            data={prepareRadarChartData()}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                r: {
                  beginAtZero: true,
                  min: 0,
                  max: 1,
                  ticks: {
                    stepSize: 0.2,
                    showLabelBackdrop: false,
                    font: {
                      size: 10
                    }
                  },
                  pointLabels: {
                    font: {
                      size: 12
                    }
                  }
                }
              },
              plugins: {
                legend: {
                  position: 'top',
                },
                title: {
                  display: true,
                  text: 'Model Performance Comparison (All Metrics)',
                  font: {
                    size: 16
                  }
                },
                tooltip: {
                  callbacks: {
                    label: function(context) {
                      return `${context.dataset.label}: ${context.raw.toFixed(3)}`;
                    },
                    afterLabel: function(context) {
                      const model = performanceData.data[context.datasetIndex];
                      const performanceMetrics = performanceData.metrics.filter(m => 
                        m.id.startsWith('avg_') && 
                        !['avg_prompt_tokens', 'avg_completion_tokens', 'avg_total_tokens'].includes(m.id)
                      );
                      const metricId = performanceMetrics[context.dataIndex]?.id;
                      if (!metricId) return '';
                      const stdDevField = getStdDevField(metricId);
                      const stdDev = model[stdDevField];
                      return `Standard Deviation: ${formatStdDev(stdDev)}`;
                    }
                  }
                }
              }
            }}
          />
        )}
      </div>
      
      <div className="mt-6">
        <h3 className="text-lg font-semibold mb-3">Data Table</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white rounded-lg overflow-hidden">
            <thead className="bg-gray-100">
              <tr>
                <th className="px-4 py-2 text-left text-sm font-semibold text-gray-600">Model</th>
                {performanceData.metrics
                  .filter(m => m.id.startsWith('avg_') && !['avg_prompt_tokens', 'avg_completion_tokens', 'avg_total_tokens'].includes(m.id))
                  .map(metric => (
                    <th key={metric.id} className="px-4 py-2 text-left text-sm font-semibold text-gray-600">
                      {metric.name}
                    </th>
                  ))}
                <th className="px-4 py-2 text-left text-sm font-semibold text-gray-600">Prompt Tokens</th>
                <th className="px-4 py-2 text-left text-sm font-semibold text-gray-600">Completion Tokens</th>
                <th className="px-4 py-2 text-left text-sm font-semibold text-gray-600">Total Tokens</th>
                <th className="px-4 py-2 text-left text-sm font-semibold text-gray-600">Evaluated Query Count</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {performanceData.data.map((model, idx) => (
                <tr key={model.model_id} className={idx % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                  <td className="px-4 py-2 text-sm text-gray-900 font-medium">{model.model_name.split('/')[1]}</td>
                  {performanceData.metrics
                    .filter(m => m.id.startsWith('avg_') && !['avg_prompt_tokens', 'avg_completion_tokens', 'avg_total_tokens'].includes(m.id))
                    .map(metric => {
                      const metricValue = model[metric.id];
                      const stdDevField = getStdDevField(metric.id);
                      const stdDev = model[stdDevField];
                      
                      return (
                        <td key={`${model.model_id}-${metric.id}`} className="px-4 py-2 text-sm text-gray-900">
                          {metricValue !== null && metricValue !== undefined ? metricValue.toFixed(3) : 'N/A'}
                          <br />
                          <span className="text-gray-500 text-xs">
                            σ: {formatStdDev(stdDev)}
                          </span>
                        </td>
                      );
                    })}
                  <td className="px-4 py-2 text-sm text-gray-900">
                    {model.avg_prompt_tokens !== null && model.avg_prompt_tokens !== undefined 
                      ? Math.round(model.avg_prompt_tokens).toLocaleString() 
                      : 'N/A'}
                  </td>
                  <td className="px-4 py-2 text-sm text-gray-900">
                    {model.avg_completion_tokens !== null && model.avg_completion_tokens !== undefined 
                      ? Math.round(model.avg_completion_tokens).toLocaleString() 
                      : 'N/A'}
                  </td>
                  <td className="px-4 py-2 text-sm text-gray-900">
                    {model.avg_total_tokens !== null && model.avg_total_tokens !== undefined 
                      ? Math.round(model.avg_total_tokens).toLocaleString() 
                      : 'N/A'}
                  </td>
                  <td className="px-4 py-2 text-sm text-gray-900">
                    {model.evaluated_query_count || 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
});

export default ModelPerformanceChart; 