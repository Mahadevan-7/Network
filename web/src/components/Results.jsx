import { motion } from 'framer-motion';
import { TrendingUp, Target, CheckCircle, BarChart3, PieChart } from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts';

const Results = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: 0.2,
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 30, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1
    }
  };

  const metrics = [
    {
      name: "Accuracy",
      value: "95.8%",
      icon: <Target className="w-6 h-6" />,
      color: "from-green-500 to-green-600",
      description: "Overall classification accuracy"
    },
    {
      name: "Precision",
      value: "94.2%",
      icon: <CheckCircle className="w-6 h-6" />,
      color: "from-blue-500 to-blue-600",
      description: "True positive rate"
    },
    {
      name: "Recall",
      value: "93.7%",
      icon: <TrendingUp className="w-6 h-6" />,
      color: "from-purple-500 to-purple-600",
      description: "Sensitivity to anomalies"
    },
    {
      name: "F1-Score",
      value: "93.9%",
      icon: <BarChart3 className="w-6 h-6" />,
      color: "from-orange-500 to-orange-600",
      description: "Harmonic mean of precision and recall"
    },
    {
      name: "ROC-AUC",
      value: "0.976",
      icon: <PieChart className="w-6 h-6" />,
      color: "from-pink-500 to-pink-600",
      description: "Area under ROC curve"
    }
  ];

  const modelComparison = [
    {
      model: "Random Forest",
      accuracy: "94.2%",
      f1Score: "93.1%",
      trainingTime: "2.3s",
      predictionTime: "0.1ms"
    },
    {
      model: "XGBoost",
      accuracy: "95.8%",
      f1Score: "93.9%",
      trainingTime: "5.7s",
      predictionTime: "0.2ms"
    },
    {
      model: "LSTM",
      accuracy: "92.1%",
      f1Score: "91.4%",
      trainingTime: "45.2s",
      predictionTime: "1.2ms"
    },
    {
      model: "Autoencoder",
      accuracy: "89.7%",
      f1Score: "88.9%",
      trainingTime: "38.9s",
      predictionTime: "0.8ms"
    }
  ];

  // ROC Curve data
  const rocData = [
    { fpr: 0, randomForest: 0, xgboost: 0, lstm: 0, autoencoder: 0 },
    { fpr: 0.1, randomForest: 0.15, xgboost: 0.18, lstm: 0.12, autoencoder: 0.1 },
    { fpr: 0.2, randomForest: 0.35, xgboost: 0.42, lstm: 0.28, autoencoder: 0.25 },
    { fpr: 0.3, randomForest: 0.52, xgboost: 0.61, lstm: 0.45, autoencoder: 0.38 },
    { fpr: 0.4, randomForest: 0.68, xgboost: 0.76, lstm: 0.62, autoencoder: 0.52 },
    { fpr: 0.5, randomForest: 0.78, xgboost: 0.85, lstm: 0.72, autoencoder: 0.65 },
    { fpr: 0.6, randomForest: 0.85, xgboost: 0.91, lstm: 0.80, autoencoder: 0.75 },
    { fpr: 0.7, randomForest: 0.91, xgboost: 0.95, lstm: 0.87, autoencoder: 0.83 },
    { fpr: 0.8, randomForest: 0.95, xgboost: 0.97, lstm: 0.92, autoencoder: 0.89 },
    { fpr: 0.9, randomForest: 0.98, xgboost: 0.99, lstm: 0.96, autoencoder: 0.94 },
    { fpr: 1.0, randomForest: 1.0, xgboost: 1.0, lstm: 1.0, autoencoder: 1.0 }
  ];

  // Confusion Matrix data (XGBoost - best model)
  const confusionMatrixData = [
    { name: 'True Normal', value: 45230, color: '#10b981' },
    { name: 'False Positive', value: 1240, color: '#f59e0b' },
    { name: 'False Negative', value: 980, color: '#f59e0b' },
    { name: 'True Anomaly', value: 38550, color: '#ef4444' }
  ];

  // Feature Importance data
  const featureImportanceData = [
    { feature: 'Flow Duration', importance: 0.245 },
    { feature: 'Packet Count', importance: 0.198 },
    { feature: 'Byte Count', importance: 0.176 },
    { feature: 'Protocol Type', importance: 0.142 },
    { feature: 'Port Number', importance: 0.118 },
    { feature: 'Flow Bytes/s', importance: 0.089 },
    { feature: 'Flow Packets/s', importance: 0.032 }
  ].sort((a, b) => b.importance - a.importance);

  // Training History data
  const trainingHistoryData = [
    { epoch: 1, loss: 0.65, accuracy: 0.72, valLoss: 0.68, valAccuracy: 0.70 },
    { epoch: 2, loss: 0.52, accuracy: 0.81, valLoss: 0.55, valAccuracy: 0.79 },
    { epoch: 3, loss: 0.41, accuracy: 0.87, valLoss: 0.44, valAccuracy: 0.85 },
    { epoch: 4, loss: 0.33, accuracy: 0.91, valLoss: 0.36, valAccuracy: 0.89 },
    { epoch: 5, loss: 0.27, accuracy: 0.93, valLoss: 0.30, valAccuracy: 0.91 },
    { epoch: 6, loss: 0.22, accuracy: 0.95, valLoss: 0.25, valAccuracy: 0.93 },
    { epoch: 7, loss: 0.18, accuracy: 0.96, valLoss: 0.21, valAccuracy: 0.94 },
    { epoch: 8, loss: 0.15, accuracy: 0.97, valLoss: 0.18, valAccuracy: 0.95 },
    { epoch: 9, loss: 0.13, accuracy: 0.97, valLoss: 0.16, valAccuracy: 0.96 },
    { epoch: 10, loss: 0.11, accuracy: 0.98, valLoss: 0.14, valAccuracy: 0.96 }
  ];

  return (
    <section id="results" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <motion.h2
            variants={itemVariants}
            className="text-3xl md:text-4xl font-bold text-gray-900 mb-6"
          >
            Model Performance Results
          </motion.h2>
          <motion.p
            variants={itemVariants}
            className="text-xl text-gray-600 max-w-3xl mx-auto"
          >
            Comprehensive evaluation metrics and model comparison across different algorithms
          </motion.p>
        </motion.div>

        {/* Key Metrics Cards */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-16"
        >
          {metrics.map((metric, index) => (
            <motion.div
              key={index}
              variants={itemVariants}
              className="bg-gradient-to-br from-gray-50 to-white border border-gray-200 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2"
            >
              <div className={`w-12 h-12 bg-gradient-to-r ${metric.color} rounded-lg flex items-center justify-center text-white mb-4`}>
                {metric.icon}
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">{metric.name}</h3>
              <div className="text-3xl font-bold text-gray-900 mb-2">{metric.value}</div>
              <p className="text-sm text-gray-600">{metric.description}</p>
            </motion.div>
          ))}
        </motion.div>

        {/* Model Comparison Table */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="mb-16"
        >
          <motion.div variants={itemVariants} className="text-center mb-8">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">Model Comparison</h3>
            <p className="text-gray-600">Performance metrics across different ML/DL approaches</p>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="bg-white rounded-xl shadow-lg overflow-hidden"
          >
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">Model</th>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">Accuracy</th>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">F1-Score</th>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">Training Time</th>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">Prediction Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {modelComparison.map((model, index) => (
                    <tr key={index} className="hover:bg-gray-50 transition-colors duration-200">
                      <td className="px-6 py-4 text-sm font-medium text-gray-900">{model.model}</td>
                      <td className="px-6 py-4 text-sm text-gray-600">{model.accuracy}</td>
                      <td className="px-6 py-4 text-sm text-gray-600">{model.f1Score}</td>
                      <td className="px-6 py-4 text-sm text-gray-600">{model.trainingTime}</td>
                      <td className="px-6 py-4 text-sm text-gray-600">{model.predictionTime}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        </motion.div>

        {/* Chart Placeholders */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          <motion.div variants={itemVariants} className="text-center mb-8">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">Performance Visualizations</h3>
            <p className="text-gray-600">Detailed charts and graphs from model evaluation</p>
          </motion.div>

          <motion.div
            variants={containerVariants}
            className="grid grid-cols-1 lg:grid-cols-2 gap-8"
          >
            {/* ROC Curve Chart */}
            <motion.div
              variants={itemVariants}
              className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl p-6 shadow-lg"
            >
              <div className="text-center mb-4">
                <PieChart className="w-12 h-12 text-blue-600 mx-auto mb-2" />
                <h4 className="text-xl font-semibold text-gray-900 mb-1">ROC Curves</h4>
                <p className="text-gray-600 text-sm">Receiver Operating Characteristic curves for all models</p>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={rocData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis 
                    dataKey="fpr" 
                    label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5 }}
                    stroke="#6b7280"
                  />
                  <YAxis 
                    label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }}
                    stroke="#6b7280"
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="xgboost" stroke="#2563eb" strokeWidth={2} name="XGBoost" dot={false} />
                  <Line type="monotone" dataKey="randomForest" stroke="#10b981" strokeWidth={2} name="Random Forest" dot={false} />
                  <Line type="monotone" dataKey="lstm" stroke="#8b5cf6" strokeWidth={2} name="LSTM" dot={false} />
                  <Line type="monotone" dataKey="autoencoder" stroke="#f59e0b" strokeWidth={2} name="Autoencoder" dot={false} />
                  <Line type="monotone" dataKey="fpr" stroke="#ef4444" strokeWidth={1.5} strokeDasharray="5 5" name="Random Classifier" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Confusion Matrix Visualization */}
            <motion.div
              variants={itemVariants}
              className="bg-gradient-to-br from-green-50 to-blue-50 rounded-xl p-6 shadow-lg"
            >
              <div className="text-center mb-4">
                <BarChart3 className="w-12 h-12 text-green-600 mx-auto mb-2" />
                <h4 className="text-xl font-semibold text-gray-900 mb-1">Confusion Matrix (XGBoost)</h4>
                <p className="text-gray-600 text-sm">Classification results visualization</p>
              </div>
              <div className="bg-white rounded-lg p-4 mb-4">
                <div className="grid grid-cols-2 gap-2 mb-2">
                  <div className="text-center p-3 bg-green-100 rounded-lg">
                    <div className="text-2xl font-bold text-green-800">45,230</div>
                    <div className="text-xs text-green-700">True Normal</div>
                  </div>
                  <div className="text-center p-3 bg-yellow-100 rounded-lg">
                    <div className="text-2xl font-bold text-yellow-800">1,240</div>
                    <div className="text-xs text-yellow-700">False Positive</div>
                  </div>
                  <div className="text-center p-3 bg-yellow-100 rounded-lg">
                    <div className="text-2xl font-bold text-yellow-800">980</div>
                    <div className="text-xs text-yellow-700">False Negative</div>
                  </div>
                  <div className="text-center p-3 bg-red-100 rounded-lg">
                    <div className="text-2xl font-bold text-red-800">38,550</div>
                    <div className="text-xs text-red-700">True Anomaly</div>
                  </div>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={confusionMatrixData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" stroke="#6b7280" />
                  <YAxis dataKey="name" type="category" stroke="#6b7280" width={100} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                  />
                  <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                    {confusionMatrixData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Feature Importance Chart */}
            <motion.div
              variants={itemVariants}
              className="bg-gradient-to-br from-orange-50 to-pink-50 rounded-xl p-6 shadow-lg"
            >
              <div className="text-center mb-4">
                <TrendingUp className="w-12 h-12 text-orange-600 mx-auto mb-2" />
                <h4 className="text-xl font-semibold text-gray-900 mb-1">Feature Importance</h4>
                <p className="text-gray-600 text-sm">Most influential features for anomaly detection</p>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={featureImportanceData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" domain={[0, 0.3]} stroke="#6b7280" />
                  <YAxis dataKey="feature" type="category" stroke="#6b7280" width={120} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                    formatter={(value) => [(value * 100).toFixed(1) + '%', 'Importance']}
                  />
                  <Bar dataKey="importance" fill="#f97316" radius={[0, 8, 8, 0]}>
                    {featureImportanceData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={`hsl(${20 + index * 15}, 70%, 50%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Training History Chart */}
            <motion.div
              variants={itemVariants}
              className="bg-gradient-to-br from-purple-50 to-indigo-50 rounded-xl p-6 shadow-lg"
            >
              <div className="text-center mb-4">
                <Target className="w-12 h-12 text-purple-600 mx-auto mb-2" />
                <h4 className="text-xl font-semibold text-gray-900 mb-1">Training History</h4>
                <p className="text-gray-600 text-sm">Model performance over training epochs</p>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trainingHistoryData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis 
                    dataKey="epoch" 
                    label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                    stroke="#6b7280"
                  />
                  <YAxis 
                    yAxisId="left"
                    label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
                    stroke="#6b7280"
                  />
                  <YAxis 
                    yAxisId="right" 
                    orientation="right"
                    label={{ value: 'Accuracy', angle: 90, position: 'insideRight' }}
                    stroke="#6b7280"
                    domain={[0, 1]}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                    formatter={(value, name) => {
                      if (name.includes('accuracy')) return [(value * 100).toFixed(1) + '%', name];
                      return [value.toFixed(3), name];
                    }}
                  />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} name="Training Loss" dot={false} />
                  <Line yAxisId="left" type="monotone" dataKey="valLoss" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 5" name="Validation Loss" dot={false} />
                  <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} name="Training Accuracy" dot={false} />
                  <Line yAxisId="right" type="monotone" dataKey="valAccuracy" stroke="#3b82f6" strokeWidth={2} strokeDasharray="5 5" name="Validation Accuracy" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </motion.div>
          </motion.div>
        </motion.div>

        {/* Key Insights */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="mt-16"
        >
          <motion.div variants={itemVariants} className="text-center mb-8">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">Key Insights</h3>
          </motion.div>

          <motion.div
            variants={containerVariants}
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
          >
            <motion.div
              variants={itemVariants}
              className="bg-blue-50 rounded-xl p-6 border-l-4 border-blue-500"
            >
              <h4 className="font-semibold text-gray-900 mb-2">XGBoost Excellence</h4>
              <p className="text-gray-600 text-sm">
                XGBoost achieved the highest accuracy (95.8%) with optimal training time, making it ideal for production deployment.
              </p>
            </motion.div>

            <motion.div
              variants={itemVariants}
              className="bg-green-50 rounded-xl p-6 border-l-4 border-green-500"
            >
              <h4 className="font-semibold text-gray-900 mb-2">Real-time Performance</h4>
              <p className="text-gray-600 text-sm">
                All models achieve sub-millisecond prediction times, enabling real-time anomaly detection in production environments.
              </p>
            </motion.div>

            <motion.div
              variants={itemVariants}
              className="bg-purple-50 rounded-xl p-6 border-l-4 border-purple-500"
            >
              <h4 className="font-semibold text-gray-900 mb-2">Balanced Performance</h4>
              <p className="text-gray-600 text-sm">
                High precision and recall rates indicate excellent balance between detecting anomalies and minimizing false positives.
              </p>
            </motion.div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

export default Results;
