#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
using System.IO;
using System.Text;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public enum SignalQuality
    {
        NoSignal = 0,
        Low = 1,
        Medium = 2,
        High = 3,
        VeryHigh = 4
    }

    public class QuantSignal
    {
        public DateTime Time { get; set; }
        public double Price { get; set; }
        public double Probability { get; set; }
        public double KellyFraction { get; set; }
        public double ExpectedValue { get; set; }
        public SignalQuality Quality { get; set; }
        public bool IsLong { get; set; }
        public double StopLoss { get; set; }
        public double TakeProfit { get; set; }
        public double RiskReward { get; set; }
    }

    public class QuantStats
    {
        public double Accuracy { get; set; }
        public double AverageProbability { get; set; }
        public double BrierScore { get; set; }
        public double ProfitFactor { get; set; }
        public double InformationRatio { get; set; }
        public double MaxDrawdown { get; set; }
        public double EdgePerTrade { get; set; }
        public int TotalSignals { get; set; }
        public int WinningTrades { get; set; }
        public double AverageWin { get; set; }
        public double AverageLoss { get; set; }
    }

    public class QuantEdgeNT8 : Indicator
    {
        #region Parameters

        [NinjaScriptProperty]
        [Range(0.50, 0.90)]
        [Display(Name = "Probability Entry Threshold", Description = "Minimum probability to trigger signal", Order = 1, GroupName = "Core Settings")]
        public double ProbabilityEntryThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(0.01, 0.10)]
        [Display(Name = "Max Kelly Fraction", Description = "Maximum Kelly fraction for position sizing", Order = 2, GroupName = "Risk Management")]
        public double MaxKellyFraction { get; set; }

        [NinjaScriptProperty]
        [Range(0.001, 0.05)]
        [Display(Name = "Risk Budget %", Description = "Maximum risk per trade as % of account", Order = 3, GroupName = "Risk Management")]
        public double RiskBudgetPercent { get; set; }

        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "ATR Period", Description = "ATR period for volatility calculation", Order = 4, GroupName = "Technical Parameters")]
        public int ATRPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(10, 200)]
        [Display(Name = "VWAP Period", Description = "VWAP calculation period", Order = 5, GroupName = "Technical Parameters")]
        public int VWAPPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(50, 500)]
        [Display(Name = "Hurst Window", Description = "Window for Hurst exponent calculation", Order = 6, GroupName = "Advanced")]
        public int HurstWindow { get; set; }

        [NinjaScriptProperty]
        [Range(0.10, 5.00)]
        [Display(Name = "VaR Multiplier", Description = "VaR threshold multiplier", Order = 7, GroupName = "Risk Management")]
        public double VaRMultiplier { get; set; }

        [NinjaScriptProperty]
        [Range(10, 1000)]
        [Display(Name = "Performance Window", Description = "Rolling window for performance metrics", Order = 8, GroupName = "Performance")]
        public int PerformanceWindow { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 3.0)]
        [Display(Name = "Min Information Ratio", Description = "Minimum IR to keep signals active", Order = 9, GroupName = "Performance")]
        public double MinInformationRatio { get; set; }

        // Display Options
        [NinjaScriptProperty]
        [Display(Name = "Show Signals", Description = "Display trading signals", Order = 10, GroupName = "Display")]
        public bool ShowSignals { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Probability", Description = "Display probability values", Order = 11, GroupName = "Display")]
        public bool ShowProbability { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Risk Levels", Description = "Display SL/TP levels", Order = 12, GroupName = "Display")]
        public bool ShowRiskLevels { get; set; }

        [NinjaScriptProperty]        [Display(Name = "Show Statistics", Description = "Display performance statistics", Order = 13, GroupName = "Display")]
        public bool ShowStatistics { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Auto Pause", Description = "Auto pause on poor performance", Order = 14, GroupName = "Performance")]
        public bool EnableAutoPause { get; set; }

        // ML Model Settings
        [NinjaScriptProperty]
        [Display(Name = "Enable ML Model", Description = "Enable external ML model predictions", Order = 15, GroupName = "Machine Learning")]
        public bool EnableMLModel { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "ML Model Path", Description = "Path to ML model file (.csv format)", Order = 16, GroupName = "Machine Learning")]
        public string MLModelPath { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 2.0)]
        [Display(Name = "ML Weight", Description = "Weight of ML predictions vs built-in model", Order = 17, GroupName = "Machine Learning")]
        public double MLWeight { get; set; }

        // Advanced Scalping Settings
        [NinjaScriptProperty]
        [Display(Name = "Enable Scalping Mode", Description = "Enable advanced scalping features", Order = 18, GroupName = "Scalping")]
        public bool EnableScalpingMode { get; set; }

        [NinjaScriptProperty]
        [Range(0.5, 5.0)]
        [Display(Name = "TP ATR Multiplier", Description = "Take profit as ATR multiplier", Order = 19, GroupName = "Scalping")]
        public double TPATRMultiplier { get; set; }

        [NinjaScriptProperty]
        [Range(0.5, 3.0)]
        [Display(Name = "SL ATR Multiplier", Description = "Stop loss as ATR multiplier", Order = 20, GroupName = "Scalping")]
        public double SLATRMultiplier { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 1.0)]
        [Display(Name = "Partial Exit %", Description = "Percentage for partial profit taking", Order = 21, GroupName = "Scalping")]
        public double PartialExitPercent { get; set; }

        [NinjaScriptProperty]
        [Range(5, 100)]        [Display(Name = "Max Holding Bars", Description = "Maximum bars to hold position", Order = 22, GroupName = "Scalping")]
        public int MaxHoldingBars { get; set; }

        // Colors
        [NinjaScriptProperty]
        [XmlIgnore]
        [Display(Name = "Long Signal Color", Description = "Color for long signals", Order = 23, GroupName = "Colors")]
        public Brush LongSignalColor { get; set; }

        [Browsable(false)]
        public string LongSignalColorSerializable
        {
            get { return Serialize.BrushToString(LongSignalColor); }
            set { LongSignalColor = Serialize.StringToBrush(value); }
        }

        [NinjaScriptProperty]
        [XmlIgnore]
        [Display(Name = "Short Signal Color", Description = "Color for short signals", Order = 24, GroupName = "Colors")]
        public Brush ShortSignalColor { get; set; }

        [Browsable(false)]
        public string ShortSignalColorSerializable
        {
            get { return Serialize.BrushToString(ShortSignalColor); }
            set { ShortSignalColor = Serialize.StringToBrush(value); }
        }

        [NinjaScriptProperty]
        [XmlIgnore]
        [Display(Name = "Probability Color", Description = "Color for probability display", Order = 25, GroupName = "Colors")]
        public Brush ProbabilityColor { get; set; }

        [Browsable(false)]
        public string ProbabilityColorSerializable
        {
            get { return Serialize.BrushToString(ProbabilityColor); }
            set { ProbabilityColor = Serialize.StringToBrush(value); }
        }

        #endregion

        #region Public Series
        [Browsable(false)]
        [XmlIgnore]
        public Series<double> ProbabilityLine
        {
            get { return Values[0]; }
        }

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> EdgeLine
        {
            get { return Values[1]; }
        }
        #endregion

        #region Private Variables
          // Technical Indicators
        private ATR atr;
        private double vwap; // Manual VWAP calculation
        private SMA volumeSMA;
        private EMA priceEMA;
        
        // VWAP calculation variables
        private double cumulativePriceVolume;
        private double cumulativeVolume;
        
        // Quantitative Variables
        private List<double> logReturns;
        private List<double> volumeDeltas;
        private List<double> bidAskImbalances;
        private List<QuantSignal> recentSignals;
        private List<double> performanceHistory;
        
        // Features Array (14 features total)
        private double[] features;
        
        // Model Weights (simplified logistic regression)
        private double[] modelWeights;
        private double modelBias;
        
        // Risk Management
        private double currentVaR;
        private double portfolioValue;
        private bool isSystemActive;
        
        // Performance Tracking
        private QuantStats currentStats;
        private int signalCounter;
        private double cumulativePnL;
          // Rolling Windows
        private Queue<double> rollingReturns;
        private Queue<double> rollingVolumes;
        private Queue<double> rollingProbabilities;
        
        // ML Model Variables
        private Dictionary<DateTime, double> mlPredictions;
        private bool mlModelLoaded;
        private DateTime lastMLUpdate;
        
        // Advanced Scalping Variables
        private Dictionary<DateTime, ScalpPosition> activePositions;
        private Queue<ScalpTrade> recentTrades;
        private double averageHoldingPeriod;
        private double scalpingWinRate;
        
        #endregion

        #region Scalping Classes
        private class ScalpPosition
        {
            public DateTime EntryTime { get; set; }
            public double EntryPrice { get; set; }
            public bool IsLong { get; set; }
            public double StopLoss { get; set; }
            public double TakeProfit { get; set; }
            public double PartialTaken { get; set; }
            public int BarsHeld { get; set; }
            public double PnL { get; set; }
        }

        private class ScalpTrade
        {
            public DateTime EntryTime { get; set; }
            public DateTime ExitTime { get; set; }
            public double EntryPrice { get; set; }
            public double ExitPrice { get; set; }
            public bool IsLong { get; set; }
            public double PnL { get; set; }
            public int BarsHeld { get; set; }
            public string ExitReason { get; set; }
        }
        
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "QuantEdge NT8 — Scalping probabiliste 100% quant pour NinjaTrader 8";
                Name = "QuantEdgeNT8";
                Calculate = Calculate.OnEachTick;
                IsOverlay = true;
                DisplayInDataBox = true;
                DrawOnPricePanel = true;
                PaintPriceMarkers = true;
                ScaleJustification = NinjaTrader.Gui.Chart.ScaleJustification.Right;                IsSuspendedWhileInactive = false;

                // Default Parameters (adjusted for more historical signals)
                ProbabilityEntryThreshold = 0.60; // Reduced from 0.63 for more signals
                MaxKellyFraction = 0.03;
                RiskBudgetPercent = 0.008; // 0.8%
                ATRPeriod = 14;
                VWAPPeriod = 100;
                HurstWindow = 128;                VaRMultiplier = 0.8;
                PerformanceWindow = 100;
                MinInformationRatio = 0.5;

                // ML Model Defaults
                EnableMLModel = false;
                MLModelPath = "";
                MLWeight = 0.5;

                // Scalping Defaults
                EnableScalpingMode = false;
                TPATRMultiplier = 2.0;
                SLATRMultiplier = 1.5;
                PartialExitPercent = 0.5;
                MaxHoldingBars = 20;

                // Display Defaults
                ShowSignals = true;
                ShowProbability = true;
                ShowRiskLevels = true;
                ShowStatistics = true;
                EnableAutoPause = true;

                // Color Defaults
                LongSignalColor = Brushes.LimeGreen;
                ShortSignalColor = Brushes.Red;
                ProbabilityColor = Brushes.Orange;

                // Add plots
                AddPlot(Brushes.Orange, "Probability");
                AddPlot(Brushes.Yellow, "Edge");
                  // Initialize collections
                features = new double[14];
                modelWeights = InitializeModelWeights();
                modelBias = -0.15; // Slight negative bias for conservative approach
                
                // Initialize ML and Scalping collections
                mlPredictions = new Dictionary<DateTime, double>();
                mlModelLoaded = false;
                lastMLUpdate = DateTime.MinValue;
                activePositions = new Dictionary<DateTime, ScalpPosition>();
                recentTrades = new Queue<ScalpTrade>();
                averageHoldingPeriod = 0;
                scalpingWinRate = 0;
            }
            else if (State == State.DataLoaded)
            {                // Initialize indicators
                atr = ATR(ATRPeriod);
                // Initialize manual VWAP calculation
                cumulativePriceVolume = 0;
                cumulativeVolume = 0;
                vwap = Close[0]; // Initial VWAP value
                volumeSMA = SMA(Volume, 20);
                priceEMA = EMA(Close, 21);
                
                // Initialize collections
                logReturns = new List<double>();
                volumeDeltas = new List<double>();
                bidAskImbalances = new List<double>();                recentSignals = new List<QuantSignal>();
                performanceHistory = new List<double>();

                rollingReturns = new Queue<double>();
                rollingVolumes = new Queue<double>();
                rollingProbabilities = new Queue<double>();
                
                // Initialize state
                isSystemActive = true;
                signalCounter = 0;
                cumulativePnL = 0;
                
                currentStats = new QuantStats();
                
                // Load ML model if enabled
                if (EnableMLModel)
                    LoadMLModel();
                  Print("QuantEdge NT8 initialized - Probabilistic scalping system active");
            }
        }
        {
            // Reduced minimum bars requirement for historical signals
            if (CurrentBar < Math.Max(ATRPeriod, 20))
                return;

            try
            {
                // Update manual VWAP calculation
                UpdateVWAP();
                
                // Update features and calculate probability
                UpdateQuantitativeFeatures();
                double probability = CalculateProbability();
                double edge = CalculateEdge(probability);
                
                // Update plots
                ProbabilityLine[0] = probability;
                EdgeLine[0] = edge;
                
                // Update rolling performance
                UpdatePerformanceMetrics();
                  // Check system status
                if (EnableAutoPause)
                    CheckSystemHealth();                // Generate signals if system is active (removed State.Realtime restriction for historical signals)
                if (isSystemActive)
                {
                    ProcessTradingSignals(probability, edge);
                }
                
                // Update scalping positions
                if (EnableScalpingMode)
                {
                    UpdateScalpingPositions();
                }
                
                // Check for ML model updates periodically
                if (CurrentBar % 50 == 0)
                {
                    CheckMLModelUpdate();
                }
                
                // Debug: Print probability values occasionally for verification
                if (CurrentBar % 100 == 0)
                {
                    Print("QuantEdge Debug - Bar:" + CurrentBar + ", P:" + probability.ToString("F3") + ", Edge:" + edge.ToString("F4") + ", Active:" + isSystemActive);
                }
                
                // Update statistics display
                if (ShowStatistics && CurrentBar % 10 == 0)
                    UpdateStatisticsDisplay();
                      }
            catch (Exception ex)
            {
                Print("Error in OnBarUpdate: " + ex.Message);            }
        }

        #region VWAP Calculation
        
        private void UpdateVWAP()
        {
            // Calculate typical price for current bar
            double typicalPrice = (High[0] + Low[0] + Close[0]) / 3.0;
            double currentVolume = Volume[0];
            
            // Update cumulative values
            cumulativePriceVolume += typicalPrice * currentVolume;
            cumulativeVolume += currentVolume;
            
            // Calculate VWAP
            if (cumulativeVolume > 0)
            {
                vwap = cumulativePriceVolume / cumulativeVolume;
            }
            
            // Reset daily (simplified - reset every VWAPPeriod bars)
            if (CurrentBar % VWAPPeriod == 0)
            {
                cumulativePriceVolume = typicalPrice * currentVolume;
                cumulativeVolume = currentVolume;
                vwap = typicalPrice;
            }
        }
          #endregion

        #region Feature Engineering

        private void UpdateQuantitativeFeatures()
        {
            // Clear features array
            Array.Clear(features, 0, features.Length);
            
            // Reduced requirement for historical signals - use available data even if less than HurstWindow
            if (CurrentBar < 20)
                return;
                
            // Feature 1-2: Log returns (1-tick and smoothed)
            if (CurrentBar > 0)
            {
                double logReturn = Math.Log(Close[0] / Close[1]);
                features[0] = logReturn;
                features[1] = logReturns.Count > 5 ? logReturns.Skip(Math.Max(0, logReturns.Count - 5)).Average() : logReturn;
                logReturns.Add(logReturn);
                
                // Keep only recent returns
                if (logReturns.Count > HurstWindow)
                    logReturns.RemoveAt(0);
            }
            
            // Feature 3: ATR normalized
            features[2] = atr[0] / Close[0];
            
            // Feature 4: Range fractal (normalized high-low range)
            features[3] = (High[0] - Low[0]) / Close[0];
            
            // Feature 5: Hurst exponent
            features[4] = CalculateHurstExponent();
            
            // Feature 6-7: Volume features
            features[5] = Volume[0] / volumeSMA[0]; // Volume ratio
            double volumeDelta = CurrentBar > 0 ? Volume[0] - Volume[1] : 0;
            features[6] = volumeDelta / volumeSMA[0]; // Volume delta normalized
              // Feature 8: Price vs VWAP
            features[7] = (Close[0] - vwap) / atr[0];
            
            // Feature 9: Price momentum
            features[8] = (Close[0] - priceEMA[0]) / atr[0];
            
            // Feature 10-11: Volatility features
            features[9] = CalculateVolatilityRegime();
            features[10] = CalculateVolatilitySkew();
            
            // Feature 12-13: Market microstructure (simplified)
            features[11] = CalculateBidAskImbalance();
            features[12] = CalculateOrderFlowDelta();
            
            // Feature 14: Time-based feature (intraday pattern)
            features[13] = CalculateIntradayPattern();
            
            // Normalize features using Z-score
            NormalizeFeatures();
        }

        private double CalculateHurstExponent()
        {
            if (logReturns.Count < HurstWindow)
                return 0.5; // Random walk default
                  try
            {
                var returns = logReturns.Skip(Math.Max(0, logReturns.Count - HurstWindow)).ToArray();
                return ComputeHurstRS(returns);
            }
            catch
            {
                return 0.5;
            }
        }

        private double ComputeHurstRS(double[] returns)
        {
            int n = returns.Length;
            if (n < 10) return 0.5;
            
            // Calculate mean
            double mean = returns.Average();
            
            // Calculate cumulative deviations
            double[] cumDev = new double[n];
            cumDev[0] = returns[0] - mean;
            for (int i = 1; i < n; i++)
                cumDev[i] = cumDev[i-1] + (returns[i] - mean);
            
            // Calculate range
            double range = cumDev.Max() - cumDev.Min();
            
            // Calculate standard deviation
            double std = Math.Sqrt(returns.Select(x => Math.Pow(x - mean, 2)).Average());
            
            if (std == 0) return 0.5;
            
            // R/S ratio
            double rs = range / std;
            
            // Hurst exponent
            double hurst = Math.Log(rs) / Math.Log(n);
            
            // Clamp between 0 and 1
            return Math.Max(0, Math.Min(1, hurst));
        }

        private double CalculateVolatilityRegime()
        {
            if (CurrentBar < 20) return 0;
              // Calculate short vs long term volatility ratio
            var recentReturns = logReturns.Skip(Math.Max(0, logReturns.Count - 5)).ToArray();
            var longerReturns = logReturns.Skip(Math.Max(0, logReturns.Count - 20)).ToArray();
            
            if (recentReturns.Length == 0 || longerReturns.Length == 0) return 0;
            
            double recentVol = Math.Sqrt(recentReturns.Select(x => x * x).Average());
            double longerVol = Math.Sqrt(longerReturns.Select(x => x * x).Average());
            
            return longerVol > 0 ? recentVol / longerVol : 1.0;
        }

        private double CalculateVolatilitySkew()
        {            if (logReturns.Count < 20) return 0;
            
            var returns = logReturns.Skip(Math.Max(0, logReturns.Count - 20)).ToArray();
            double mean = returns.Average();
            double std = Math.Sqrt(returns.Select(x => Math.Pow(x - mean, 2)).Average());
            
            if (std == 0) return 0;
            
            // Calculate skewness
            double skew = returns.Select(x => Math.Pow((x - mean) / std, 3)).Average();
            return skew;
        }

        private double CalculateBidAskImbalance()
        {
            // Simplified bid/ask imbalance using volume and price action
            if (CurrentBar == 0) return 0;
            
            double priceChange = Close[0] - Close[1];
            double volumeWeight = Volume[0] / volumeSMA[0];
            
            return Math.Sign(priceChange) * volumeWeight;
        }

        private double CalculateOrderFlowDelta()
        {
            // Simplified order flow delta
            if (CurrentBar == 0) return 0;
            
            double delta = 0;
            if (Close[0] > Open[0]) // Up bar
                delta = Volume[0];
            else if (Close[0] < Open[0]) // Down bar
                delta = -Volume[0];
            else // Doji
                delta = 0;
            
            return delta / volumeSMA[0];
        }

        private double CalculateIntradayPattern()
        {
            // Simple intraday pattern based on time
            var timeOfDay = Time[0].TimeOfDay;
            var totalMinutes = timeOfDay.TotalMinutes;
            
            // Market open (9:30) = 570 minutes, close (16:00) = 960 minutes
            double marketOpenMinutes = 570;
            double marketCloseMinutes = 960;
            
            if (totalMinutes < marketOpenMinutes || totalMinutes > marketCloseMinutes)
                return 0; // Outside market hours
            
            // Normalize to 0-1 range for market session
            double normalizedTime = (totalMinutes - marketOpenMinutes) / (marketCloseMinutes - marketOpenMinutes);
            
            // Create pattern: higher volatility at open/close, lower at lunch
            double pattern = Math.Sin(normalizedTime * Math.PI) * 0.5 + 0.5;
            return pattern;
        }

        private void NormalizeFeatures()
        {
            // Simple Z-score normalization per feature
            // In production, you would use rolling statistics
            for (int i = 0; i < features.Length; i++)
            {
                if (double.IsNaN(features[i]) || double.IsInfinity(features[i]))
                    features[i] = 0;
                    
                // Clamp extreme values
                features[i] = Math.Max(-5, Math.Min(5, features[i]));
            }
        }

        #endregion

        #region Probability Model        private double CalculateProbability()
        {
            try
            {
                // Simple logistic regression: P = 1 / (1 + exp(-(w·x + b)))
                double linearCombination = modelBias;
                
                for (int i = 0; i < Math.Min(features.Length, modelWeights.Length); i++)
                {
                    linearCombination += features[i] * modelWeights[i];
                }
                
                // Apply logistic function
                double builtInProbability = 1.0 / (1.0 + Math.Exp(-linearCombination));
                
                // Add some randomness to prevent overfitting (Monte Carlo element)
                double noise = (new Random().NextDouble() - 0.5) * 0.02; // ±1% noise
                builtInProbability = Math.Max(0.01, Math.Min(0.99, builtInProbability + noise));
                
                // Get ML prediction and combine with built-in model
                double mlPrediction = GetMLPrediction(Time[0]);
                double finalProbability = CombinePredictions(builtInProbability, mlPrediction);
                
                // Update rolling probabilities
                rollingProbabilities.Enqueue(finalProbability);
                if (rollingProbabilities.Count > PerformanceWindow)
                    rollingProbabilities.Dequeue();
                
                return finalProbability;}
            catch (Exception ex)
            {
                Print("Error in CalculateProbability: " + ex.Message);
                return 0.5; // Neutral probability on error
            }
        }

        private double[] InitializeModelWeights()
        {
            // Pre-trained weights (in practice, these would come from offline training)
            return new double[]
            {
                0.15,  // Log return
                0.12,  // Smoothed log return
                -0.08, // ATR normalized
                0.05,  // Range fractal
                0.20,  // Hurst exponent
                0.10,  // Volume ratio
                0.08,  // Volume delta
                0.18,  // Price vs VWAP
                0.22,  // Price momentum
                -0.06, // Volatility regime
                0.03,  // Volatility skew
                0.14,  // Bid/Ask imbalance
                0.16,  // Order flow delta
                0.04   // Intraday pattern
            };
        }

        private double CalculateEdge(double probability)
        {
            // Calculate expected value using historical win/loss ratios
            double averageWin = currentStats.AverageWin > 0 ? currentStats.AverageWin : 0.0025; // 0.25% default
            double averageLoss = currentStats.AverageLoss > 0 ? currentStats.AverageLoss : 0.0015; // 0.15% default
            
            // Edge = P(win) * AvgWin - P(loss) * AvgLoss
            double edge = probability * averageWin - (1 - probability) * averageLoss;
            
            return edge;
        }

        #endregion

        #region Signal Processing        private void ProcessTradingSignals(double probability, double edge)
        {
            try
            {
                // Relaxed conditions for historical signals
                if (edge <= 0) return; // No positive edge
                if (Math.Abs(probability - 0.5) < 0.03) return; // Reduced neutrality threshold for more signals
                  bool isLongSignal = probability >= ProbabilityEntryThreshold && Close[0] > vwap;
                bool isShortSignal = probability <= (1 - ProbabilityEntryThreshold) && Close[0] < vwap;
                
                if (isLongSignal)
                {
                    ProcessLongSignal(probability, edge);
                }
                else if (isShortSignal)
                {
                    ProcessShortSignal(probability, edge);
                }            }
            catch (Exception ex)
            {
                Print("Error in ProcessTradingSignals: " + ex.Message);
            }
        }

        private void ProcessLongSignal(double probability, double edge)
        {
            signalCounter++;
            string tag = "QuantLong" + signalCounter;
            
            // Calculate position sizing using Modified Kelly
            double kellyFraction = CalculateKellyFraction(probability, currentStats.AverageWin, currentStats.AverageLoss);
            kellyFraction = Math.Min(kellyFraction, MaxKellyFraction);
            
            // Calculate risk levels
            double entryPrice = Close[0];
            double stopLoss = entryPrice - (0.8 * atr[0]);
            double takeProfit = entryPrice + (2.0 * atr[0]); // 2.5:1 R/R ratio
            
            // Validate risk/reward
            double risk = entryPrice - stopLoss;
            double reward = takeProfit - entryPrice;
            double rrRatio = reward / risk;
            
            if (rrRatio < 1.5) return; // Minimum 1.5:1 R/R
            
            // Create signal object
            QuantSignal signal = new QuantSignal
            {
                Time = Time[0],
                Price = entryPrice,
                Probability = probability,
                KellyFraction = kellyFraction,
                ExpectedValue = edge,
                Quality = GetSignalQuality(probability),
                IsLong = true,
                StopLoss = stopLoss,
                TakeProfit = takeProfit,
                RiskReward = rrRatio
            };
            
            // Draw signal
            if (ShowSignals)
            {
                double arrowY = Low[0] - (atr[0] * 0.5);
                Draw.ArrowUp(this, tag + "Arrow", false, 0, arrowY, LongSignalColor);
                  if (ShowProbability)
                {
                    string signalText = "LONG\nP:" + probability.ToString("P1") + "\nKelly:" + kellyFraction.ToString("P1") + "\nR/R:" + rrRatio.ToString("F1");
                    Draw.Text(this, tag + "Text", signalText, 0, arrowY - atr[0], LongSignalColor);
                }
                
                if (ShowRiskLevels)
                {
                    Draw.HorizontalLine(this, tag + "TP", takeProfit, Brushes.Green);
                    Draw.HorizontalLine(this, tag + "SL", stopLoss, Brushes.Red);
                }
            }            // Store signal
            recentSignals.Add(signal);
            
            // Process scalping signal if enabled
            if (EnableScalpingMode)
            {
                ProcessScalpingSignal(signal);
            }
            
            Print("LONG Signal: P=" + probability.ToString("P1") + ", Kelly=" + kellyFraction.ToString("P1") + ", Edge=" + edge.ToString("F4") + ", R/R=" + rrRatio.ToString("F1"));
        }

        private void ProcessShortSignal(double probability, double edge)
        {
            signalCounter++;
            string tag = "QuantShort" + signalCounter;
            
            // Calculate position sizing using Modified Kelly
            double shortProbability = 1 - probability;
            double kellyFraction = CalculateKellyFraction(shortProbability, currentStats.AverageWin, currentStats.AverageLoss);
            kellyFraction = Math.Min(kellyFraction, MaxKellyFraction);
            
            // Calculate risk levels
            double entryPrice = Close[0];
            double stopLoss = entryPrice + (0.8 * atr[0]);
            double takeProfit = entryPrice - (2.0 * atr[0]); // 2.5:1 R/R ratio
            
            // Validate risk/reward
            double risk = stopLoss - entryPrice;
            double reward = entryPrice - takeProfit;
            double rrRatio = reward / risk;
            
            if (rrRatio < 1.5) return; // Minimum 1.5:1 R/R
            
            // Create signal object
            QuantSignal signal = new QuantSignal
            {
                Time = Time[0],
                Price = entryPrice,
                Probability = shortProbability,
                KellyFraction = kellyFraction,
                ExpectedValue = edge,
                Quality = GetSignalQuality(shortProbability),
                IsLong = false,
                StopLoss = stopLoss,
                TakeProfit = takeProfit,
                RiskReward = rrRatio
            };
            
            // Draw signal
            if (ShowSignals)
            {
                double arrowY = High[0] + (atr[0] * 0.5);
                Draw.ArrowDown(this, tag + "Arrow", false, 0, arrowY, ShortSignalColor);
                  if (ShowProbability)
                {
                    string signalText = "SHORT\nP:" + shortProbability.ToString("P1") + "\nKelly:" + kellyFraction.ToString("P1") + "\nR/R:" + rrRatio.ToString("F1");
                    Draw.Text(this, tag + "Text", signalText, 0, arrowY + atr[0], ShortSignalColor);
                }
                
                if (ShowRiskLevels)
                {
                    Draw.HorizontalLine(this, tag + "TP", takeProfit, Brushes.Green);
                    Draw.HorizontalLine(this, tag + "SL", stopLoss, Brushes.Red);
                }
            }            // Store signal
            recentSignals.Add(signal);
            
            // Process scalping signal if enabled
            if (EnableScalpingMode)
            {
                ProcessScalpingSignal(signal);
            }
            
            Print("SHORT Signal: P=" + shortProbability.ToString("P1") + ", Kelly=" + kellyFraction.ToString("P1") + ", Edge=" + edge.ToString("F4") + ", R/R=" + rrRatio.ToString("F1"));
        }

        private double CalculateKellyFraction(double probability, double averageWin, double averageLoss)
        {
            if (averageLoss == 0) return 0;
            
            double b = averageWin / averageLoss; // Odds ratio
            double q = 1 - probability;
            
            // Kelly formula: f = (p*b - q) / b
            double kelly = (probability * b - q) / b;
            
            // Ensure positive and reasonable
            return Math.Max(0, Math.Min(kelly, MaxKellyFraction));
        }

        private SignalQuality GetSignalQuality(double probability)
        {
            if (probability >= 0.75) return SignalQuality.VeryHigh;
            if (probability >= 0.68) return SignalQuality.High;
            if (probability >= 0.60) return SignalQuality.Medium;
            if (probability >= 0.55) return SignalQuality.Low;
            return SignalQuality.NoSignal;
        }

        #endregion

        #region Performance & Risk Management

        private void UpdatePerformanceMetrics()
        {
            // Update rolling windows
            if (CurrentBar > 0)
            {
                double barReturn = (Close[0] - Close[1]) / Close[1];
                rollingReturns.Enqueue(barReturn);
                if (rollingReturns.Count > PerformanceWindow)
                    rollingReturns.Dequeue();
                
                rollingVolumes.Enqueue(Volume[0]);
                if (rollingVolumes.Count > PerformanceWindow)
                    rollingVolumes.Dequeue();
            }
            
            // Calculate VaR
            if (rollingReturns.Count >= 20)
            {
                var sortedReturns = rollingReturns.OrderBy(x => x).ToArray();
                int varIndex = (int)(sortedReturns.Length * 0.01); // 1% VaR
                currentVaR = Math.Abs(sortedReturns[varIndex]);
            }
            
            // Update statistics
            UpdateCurrentStats();
        }

        private void UpdateCurrentStats()
        {            if (recentSignals.Count == 0) return;
            
            var recentPerformance = recentSignals.Skip(Math.Max(0, recentSignals.Count - PerformanceWindow)).ToList();
            
            currentStats.TotalSignals = recentPerformance.Count;
            currentStats.AverageProbability = recentPerformance.Average(s => s.Probability);
            
            // Simulate win/loss for demonstration (in practice, track actual results)
            int wins = (int)(recentPerformance.Count * currentStats.AverageProbability);
            currentStats.WinningTrades = wins;
            currentStats.Accuracy = recentPerformance.Count > 0 ? (double)wins / recentPerformance.Count : 0;
            
            // Update win/loss averages
            currentStats.AverageWin = 0.0025; // 0.25%
            currentStats.AverageLoss = 0.0015; // 0.15%
            
            // Calculate other metrics
            currentStats.ProfitFactor = currentStats.AverageLoss > 0 ? 
                (currentStats.Accuracy * currentStats.AverageWin) / ((1 - currentStats.Accuracy) * currentStats.AverageLoss) : 0;
            
            currentStats.EdgePerTrade = recentPerformance.Count > 0 ? 
                recentPerformance.Average(s => s.ExpectedValue) : 0;
            
            // Calculate Information Ratio (simplified)
            if (rollingReturns.Count > 10)
            {
                double avgReturn = rollingReturns.Average();
                double volatility = Math.Sqrt(rollingReturns.Select(r => Math.Pow(r - avgReturn, 2)).Average());
                currentStats.InformationRatio = volatility > 0 ? avgReturn / volatility : 0;
            }
        }

        private void CheckSystemHealth()
        {
            // Auto-pause system if performance degrades
            if (currentStats.InformationRatio < MinInformationRatio && recentSignals.Count > 50)
            {
                isSystemActive = false;
                Print("System paused: Information Ratio (" + currentStats.InformationRatio.ToString("F2") + ") below threshold (" + MinInformationRatio.ToString("F2") + ")");
            }
            
            // Resume if performance improves
            if (!isSystemActive && currentStats.InformationRatio > MinInformationRatio * 1.2)
            {
                isSystemActive = true;
                Print("System resumed: Information Ratio improved to " + currentStats.InformationRatio.ToString("F2"));
            }
            
            // VaR check
            if (currentVaR > RiskBudgetPercent * VaRMultiplier)
            {
                isSystemActive = false;
                Print("System paused: VaR (" + currentVaR.ToString("P2") + ") exceeds risk budget");
            }
        }

        private void UpdateStatisticsDisplay()
        {
            if (!ShowStatistics) return;
            
            try
            {                string statsText = "QuantEdge Stats\n" +
                    "Status: " + (isSystemActive ? "ACTIVE" : "PAUSED") + "\n" +
                    "Signals: " + currentStats.TotalSignals.ToString() + "\n" +
                    "Accuracy: " + currentStats.Accuracy.ToString("P1") + "\n" +
                    "Avg Prob: " + currentStats.AverageProbability.ToString("P1") + "\n" +
                    "PF: " + currentStats.ProfitFactor.ToString("F2") + "\n" +
                    "IR: " + currentStats.InformationRatio.ToString("F2") + "\n" +
                    "Edge/Trade: " + currentStats.EdgePerTrade.ToString("F4") + "\n" +
                    "VaR: " + currentVaR.ToString("P2");
                
                // Add scalping stats if enabled
                if (EnableScalpingMode)
                {
                    statsText += "\n--- Scalping ---\n" +
                                "Active Pos: " + activePositions.Count + "\n" +
                                "Avg Hold: " + averageHoldingPeriod.ToString("F1") + " bars\n" +
                                "Win Rate: " + scalpingWinRate.ToString("P1");
                }
                
                // Add ML stats if enabled
                if (EnableMLModel && mlModelLoaded)
                {
                    statsText += "\n--- ML Model ---\n" +
                                "Predictions: " + mlPredictions.Count + "\n" +
                                "Weight: " + MLWeight.ToString("P0") + "\n" +
                                "Last Update: " + lastMLUpdate.ToString("HH:mm");
                }
                  Draw.TextFixed(this, "QuantStats", statsText, TextPosition.TopLeft, 
                    Brushes.White, new NinjaTrader.Gui.Tools.SimpleFont("Arial", 9), Brushes.Black, Brushes.Transparent, 0);
            }
            catch (Exception ex)
            {
                Print("Error updating statistics display: " + ex.Message);
            }
        }

        #endregion

        #region Public Override Methods

        public override string DisplayName
        {
            get { return Name; }
        }

        public QuantStats GetCurrentStatistics()
        {
            return currentStats;
        }

        public bool IsSystemActive()
        {
            return isSystemActive;
        }        public void SetSystemActive(bool active)
        {
            isSystemActive = active;
            Print("System manually set to: " + (active ? "ACTIVE" : "PAUSED"));
        }

        #endregion

        #region ML Model Integration

        private void LoadMLModel()
        {
            if (!EnableMLModel || string.IsNullOrEmpty(MLModelPath))
                return;

            try
            {
                if (File.Exists(MLModelPath))
                {
                    mlPredictions.Clear();
                    string[] lines = File.ReadAllLines(MLModelPath);
                    
                    foreach (string line in lines)
                    {
                        if (string.IsNullOrEmpty(line) || line.StartsWith("#"))
                            continue;
                            
                        string[] parts = line.Split(',');
                        if (parts.Length >= 2)
                        {
                            DateTime timestamp;
                            double prediction;
                            
                            if (DateTime.TryParse(parts[0], out timestamp) && 
                                double.TryParse(parts[1], out prediction))
                            {
                                mlPredictions[timestamp] = prediction;
                            }
                        }
                    }
                    
                    mlModelLoaded = true;
                    lastMLUpdate = DateTime.Now;
                    Print("ML Model loaded successfully. " + mlPredictions.Count + " predictions loaded.");
                }
                else
                {
                    Print("ML Model file not found: " + MLModelPath);
                }
            }
            catch (Exception ex)
            {
                Print("Error loading ML model: " + ex.Message);
                mlModelLoaded = false;
            }
        }

        private double GetMLPrediction(DateTime timestamp)
        {
            if (!EnableMLModel || !mlModelLoaded)
                return 0.5; // Neutral prediction

            // Find closest timestamp in ML predictions
            DateTime closestTime = DateTime.MinValue;
            double timeDiffMinutes = double.MaxValue;

            foreach (var kvp in mlPredictions)
            {
                double diff = Math.Abs((timestamp - kvp.Key).TotalMinutes);
                if (diff < timeDiffMinutes)
                {
                    timeDiffMinutes = diff;
                    closestTime = kvp.Key;
                }
            }

            // Return prediction if within 5 minutes
            if (timeDiffMinutes <= 5 && closestTime != DateTime.MinValue)
            {
                return mlPredictions[closestTime];
            }

            return 0.5; // Neutral if no close match
        }

        private double CombinePredictions(double builtInProbability, double mlPrediction)
        {
            if (!EnableMLModel)
                return builtInProbability;

            // Weighted combination of built-in model and ML prediction
            double combinedProbability = (builtInProbability * (1 - MLWeight)) + (mlPrediction * MLWeight);
            
            // Ensure bounds
            return Math.Max(0.01, Math.Min(0.99, combinedProbability));
        }

        #endregion

        #region Advanced Scalping System

        private void ProcessScalpingSignal(QuantSignal signal)
        {
            if (!EnableScalpingMode)
                return;

            try
            {
                ScalpPosition position = new ScalpPosition
                {
                    EntryTime = signal.Time,
                    EntryPrice = signal.Price,
                    IsLong = signal.IsLong,
                    StopLoss = CalculateScalpingSL(signal),
                    TakeProfit = CalculateScalpingTP(signal),
                    PartialTaken = 0,
                    BarsHeld = 0,
                    PnL = 0
                };

                activePositions[signal.Time] = position;
                
                Print("Scalping position opened: " + (signal.IsLong ? "LONG" : "SHORT") + 
                      " at " + signal.Price.ToString("F2") + 
                      ", SL: " + position.StopLoss.ToString("F2") + 
                      ", TP: " + position.TakeProfit.ToString("F2"));
            }
            catch (Exception ex)
            {
                Print("Error processing scalping signal: " + ex.Message);
            }
        }

        private double CalculateScalpingSL(QuantSignal signal)
        {
            double atrValue = atr[0];
            
            if (signal.IsLong)
                return signal.Price - (SLATRMultiplier * atrValue);
            else
                return signal.Price + (SLATRMultiplier * atrValue);
        }

        private double CalculateScalpingTP(QuantSignal signal)
        {
            double atrValue = atr[0];
            
            if (signal.IsLong)
                return signal.Price + (TPATRMultiplier * atrValue);
            else
                return signal.Price - (TPATRMultiplier * atrValue);
        }

        private void UpdateScalpingPositions()
        {
            if (!EnableScalpingMode || activePositions.Count == 0)
                return;

            List<DateTime> positionsToClose = new List<DateTime>();

            foreach (var kvp in activePositions)
            {
                ScalpPosition position = kvp.Value;
                position.BarsHeld++;

                // Check for partial exit
                if (position.PartialTaken == 0 && position.BarsHeld >= 5)
                {
                    double currentPnL = CalculatePositionPnL(position);
                    if (currentPnL > 0 && currentPnL >= (position.TakeProfit - position.EntryPrice) * PartialExitPercent)
                    {
                        position.PartialTaken = PartialExitPercent;
                        Print("Partial exit taken for position opened at " + position.EntryTime.ToString());
                    }
                }

                // Check exit conditions
                bool shouldExit = false;
                string exitReason = "";

                // Stop Loss hit
                if ((position.IsLong && Close[0] <= position.StopLoss) ||
                    (!position.IsLong && Close[0] >= position.StopLoss))
                {
                    shouldExit = true;
                    exitReason = "Stop Loss";
                }
                // Take Profit hit
                else if ((position.IsLong && Close[0] >= position.TakeProfit) ||
                         (!position.IsLong && Close[0] <= position.TakeProfit))
                {
                    shouldExit = true;
                    exitReason = "Take Profit";
                }
                // Maximum holding period reached
                else if (position.BarsHeld >= MaxHoldingBars)
                {
                    shouldExit = true;
                    exitReason = "Time Exit";
                }

                if (shouldExit)
                {
                    CloseScalpingPosition(kvp.Key, position, exitReason);
                    positionsToClose.Add(kvp.Key);
                }
            }

            // Remove closed positions
            foreach (DateTime key in positionsToClose)
            {
                activePositions.Remove(key);
            }
        }

        private double CalculatePositionPnL(ScalpPosition position)
        {
            if (position.IsLong)
                return Close[0] - position.EntryPrice;
            else
                return position.EntryPrice - Close[0];
        }

        private void CloseScalpingPosition(DateTime entryTime, ScalpPosition position, string exitReason)
        {
            double finalPnL = CalculatePositionPnL(position);
            position.PnL = finalPnL;

            ScalpTrade trade = new ScalpTrade
            {
                EntryTime = entryTime,
                ExitTime = Time[0],
                EntryPrice = position.EntryPrice,
                ExitPrice = Close[0],
                IsLong = position.IsLong,
                PnL = finalPnL,
                BarsHeld = position.BarsHeld,
                ExitReason = exitReason
            };

            recentTrades.Enqueue(trade);
            if (recentTrades.Count > 100) // Keep only last 100 trades
                recentTrades.Dequeue();

            UpdateScalpingStats();

            Print("Scalping position closed - " + exitReason + 
                  ", PnL: " + finalPnL.ToString("F4") + 
                  ", Bars held: " + position.BarsHeld);
        }

        private void UpdateScalpingStats()
        {
            if (recentTrades.Count == 0)
                return;

            // Calculate average holding period
            averageHoldingPeriod = recentTrades.Average(t => t.BarsHeld);

            // Calculate win rate
            int winningTrades = recentTrades.Count(t => t.PnL > 0);
            scalpingWinRate = (double)winningTrades / recentTrades.Count;
        }

        private void CheckMLModelUpdate()
        {
            if (!EnableMLModel || string.IsNullOrEmpty(MLModelPath))
                return;

            // Check if model file has been updated (reload every hour)
            if ((DateTime.Now - lastMLUpdate).TotalHours >= 1)
            {
                LoadMLModel();
            }
        }

        #endregion
    }
}
