package org.example.Fetch;

import java.util.*;
import java.io.*;

public class GCN {
    private final Graph graph;
    private final int dimensions;
    private final int epochs;
    private final double learningRate;

    private Map<String, double[]> embeddings;
    private double[][] W; // weight matrix (input_dim → output_dim)
    private Random random;
    private long startTime;

    // Optimized data structures
    private Map<String, double[]> initialFeatures;
    private Map<String, Double> invSqrtDegree;
    private List<String> allNodes;
    private int nodeCount;

    // Cache for performance
    private Map<String, double[]> aggregatedCache;

    public GCN(Graph graph, int dimensions, double learningRate, int epochs) {
        this.graph = graph;
        this.dimensions = dimensions;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.random = new Random(42);
        this.embeddings = new HashMap<>();
        this.initialFeatures = new HashMap<>();
        this.aggregatedCache = new HashMap<>();

        // Precompute optimized structures
        this.allNodes = new ArrayList<>(graph.getAdjacencyList().keySet());
        this.nodeCount = allNodes.size();
        precomputeDegreeInformation();
    }

    private void precomputeDegreeInformation() {
        invSqrtDegree = new HashMap<>();
        for (String node : allNodes) {
            int degree = graph.getNeighbors(node).size() + 1; // +1 for self-loop
            invSqrtDegree.put(node, 1.0 / Math.sqrt(degree));
        }
    }

    public void fit() {
        System.out.println("\n🚀 Starting OPTIMIZED GCN Training");
        System.out.println("==================================");
        System.out.printf("Graph: %d nodes, %d edges%n", graph.getVertexCount(), graph.getEdgeCount());
        System.out.printf("Parameters: dim=%d, lr=%.4f, epochs=%d%n",
                dimensions, learningRate, epochs);

        startTime = System.currentTimeMillis();

        // Phase 1: Initialize features and weights
        System.out.println("\n🎯 Phase 1: Initializing node features and weight matrix...");
        initializeFeaturesAndWeights();

        // Phase 2: Training with optimizations
        System.out.println("\n🎯 Phase 2: Training GCN model...");
        train();

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.printf("\n✅ GCN training completed in %.2f seconds%n", totalTime / 1000.0);
    }

    private void initializeFeaturesAndWeights() {
        // Use Xavier initialization for better convergence
        double xavier = Math.sqrt(6.0 / (dimensions + dimensions));

        // Initialize features with Xavier initialization
        for (String node : allNodes) {
            double[] feat = new double[dimensions];
            for (int i = 0; i < dimensions; i++) {
                feat[i] = (random.nextDouble() - 0.5) * 2.0 * xavier;
            }
            initialFeatures.put(node, feat);
        }

        // Initialize weight matrix with Xavier initialization
        W = new double[dimensions][dimensions];
        for (int i = 0; i < dimensions; i++) {
            for (int j = 0; j < dimensions; j++) {
                W[i][j] = (random.nextDouble() - 0.5) * 2.0 * xavier;
            }
        }

        System.out.println("   ✅ Features and weight matrix initialized (Xavier)");
    }

    private void train() {
        double[][] gradW = new double[dimensions][dimensions];

        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.printf("\n   📚 Epoch %d/%d%n", epoch + 1, epochs);
            long epochStartTime = System.currentTimeMillis();
            double totalLoss = 0.0;
            int processed = 0;

            // Reset gradient matrix
            for (int i = 0; i < dimensions; i++) {
                Arrays.fill(gradW[i], 0.0);
            }

            // Precompute aggregated features for this epoch
            precomputeAggregatedFeatures();

            // Step 1: Compute current embeddings via optimized propagation
            Map<String, double[]> currentEmbeddings = optimizedPropagate();

            // Step 2: Sample-based training for efficiency
            int samplesPerNode = 5; // Reduced from exhaustive sampling
            List<String> shuffledNodes = new ArrayList<>(allNodes);
            Collections.shuffle(shuffledNodes);

            for (String node : shuffledNodes) {
                Set<String> neighbors = graph.getNeighbors(node);
                if (neighbors.isEmpty()) {
                    processed++;
                    continue;
                }

                double[] selfEmb = currentEmbeddings.get(node);

                // Sample positive examples
                List<String> neighborList = new ArrayList<>(neighbors);
                Collections.shuffle(neighborList);
                int posSamples = Math.min(3, neighborList.size());

                for (int i = 0; i < posSamples; i++) {
                    String nb = neighborList.get(i);
                    double[] nbEmb = currentEmbeddings.get(nb);
                    if (nbEmb != null) {
                        totalLoss += updateGradientsOptimized(selfEmb, nbEmb, true, gradW, node, nb);
                    }
                }

                // Sample negative examples
                Set<String> nonNeighbors = new HashSet<>(allNodes);
                nonNeighbors.removeAll(neighbors);
                nonNeighbors.remove(node);

                if (!nonNeighbors.isEmpty()) {
                    List<String> nonNbList = new ArrayList<>(nonNeighbors);
                    Collections.shuffle(nonNbList);
                    int negSamples = Math.min(2, nonNbList.size());

                    for (int i = 0; i < negSamples; i++) {
                        String negNode = nonNbList.get(i);
                        double[] negEmb = currentEmbeddings.get(negNode);
                        if (negEmb != null) {
                            totalLoss += updateGradientsOptimized(selfEmb, negEmb, false, gradW, node, negNode);
                        }
                    }
                }

                processed++;
                if (processed % Math.max(1, nodeCount / 5) == 0) {
                    double progress = (double) processed / nodeCount * 100;
                    System.out.printf("      📊 Processed %d/%d nodes (%.1f%%) - Avg Loss: %.6f%n",
                            processed, nodeCount, progress, totalLoss / processed);
                }
            }

            // Update weight matrix with momentum-like effect
            updateWeightsWithMomentum(gradW, epoch);

            long epochTime = System.currentTimeMillis() - epochStartTime;
            double avgLoss = totalLoss / Math.max(1, processed);
            System.out.printf("      ✅ Epoch completed in %.2f seconds - Avg Loss: %.6f%n",
                    epochTime / 1000.0, avgLoss);
        }

        // Final embeddings
        System.out.println("\n   🔁 Computing final embeddings...");
        this.embeddings = optimizedPropagate();
        System.out.println("   ✅ Final embeddings computed");
    }

    private void precomputeAggregatedFeatures() {
        aggregatedCache.clear();

        for (String node : allNodes) {
            double[] aggregated = new double[dimensions];
            double[] selfFeat = initialFeatures.get(node);
            double selfWeight = invSqrtDegree.get(node) * invSqrtDegree.get(node);

            // Self-loop contribution
            for (int k = 0; k < dimensions; k++) {
                aggregated[k] += selfWeight * selfFeat[k];
            }

            // Neighbors contribution
            for (String nb : graph.getNeighbors(node)) {
                double[] nbFeat = initialFeatures.get(nb);
                double weight = invSqrtDegree.get(node) * invSqrtDegree.get(nb);
                for (int k = 0; k < dimensions; k++) {
                    aggregated[k] += weight * nbFeat[k];
                }
            }
            aggregatedCache.put(node, aggregated);
        }
    }

    private Map<String, double[]> optimizedPropagate() {
        Map<String, double[]> output = new HashMap<>();

        // Apply weight matrix and activation in batch using cached aggregated features
        for (String node : allNodes) {
            double[] aggregated = aggregatedCache.get(node);
            double[] newEmb = new double[dimensions];

            // Matrix multiplication: aggregated * W
            for (int i = 0; i < dimensions; i++) {
                for (int j = 0; j < dimensions; j++) {
                    newEmb[i] += aggregated[j] * W[j][i];
                }
                // ReLU activation
                newEmb[i] = Math.max(0, newEmb[i]);
            }
            output.put(node, newEmb);
        }

        return output;
    }

    private double updateGradientsOptimized(double[] emb1, double[] emb2, boolean positive,
                                            double[][] gradW, String node1, String node2) {
        // Fast dot product
        double dot = fastDotProduct(emb1, emb2);

        double label = positive ? 1.0 : 0.0;
        double sigmoid = 1.0 / (1.0 + Math.exp(-dot));
        double loss = - (label * Math.log(sigmoid + 1e-9) + (1 - label) * Math.log(1 - sigmoid + 1e-9));

        double commonGrad = (sigmoid - label);

        // Get precomputed aggregated features from cache
        double[] preAct1 = aggregatedCache.get(node1);
        double[] preAct2 = aggregatedCache.get(node2);

        // Update gradients efficiently
        for (int i = 0; i < dimensions; i++) {
            double gradEmb1_i = commonGrad * emb2[i];
            double gradEmb2_i = commonGrad * emb1[i];

            for (int j = 0; j < dimensions; j++) {
                gradW[j][i] += gradEmb1_i * preAct1[j] + gradEmb2_i * preAct2[j];
            }
        }

        return loss;
    }

    private double fastDotProduct(double[] a, double[] b) {
        double result = 0.0;
        for (int i = 0; i < dimensions; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    private void updateWeightsWithMomentum(double[][] gradW, int epoch) {
        // Simple learning rate decay
        double currentLR = learningRate * (1.0 / (1.0 + 0.1 * epoch));

        // Add L2 regularization to prevent overfitting
        double lambda = 0.001;

        for (int i = 0; i < dimensions; i++) {
            for (int j = 0; j < dimensions; j++) {
                // Gradient descent with L2 regularization
                gradW[i][j] += lambda * W[i][j];
                W[i][j] -= currentLR * gradW[i][j];
            }
        }
    }

    // Alternative: Ultra-fast GCN with 2-layer propagation
    public void fitFast() {
        System.out.println("\n🚀 Starting ULTRA-FAST GCN Training");
        System.out.println("===================================");
        System.out.printf("Graph: %d nodes, %d edges%n", graph.getVertexCount(), graph.getEdgeCount());
        System.out.printf("Parameters: dim=%d, lr=%.4f, steps=%d%n",
                dimensions, learningRate, 3);

        startTime = System.currentTimeMillis();

        // Simple initialization
        initializeFeaturesAndWeights();

        // Just do 2-3 propagation steps (no backpropagation)
        for (int step = 0; step < 3; step++) {
            System.out.printf("   🔄 Fast propagation step %d/3%n", step + 1);
            long stepStart = System.currentTimeMillis();

            // Precompute aggregated features
            precomputeAggregatedFeatures();

            // Apply propagation
            this.embeddings = optimizedPropagate();

            // Update initial features for next propagation (if any)
            if (step < 2) {
                for (String node : allNodes) {
                    initialFeatures.put(node, embeddings.get(node).clone());
                }
            }

            long stepTime = System.currentTimeMillis() - stepStart;
            System.out.printf("      ✅ Step completed in %.2f seconds%n", stepTime / 1000.0);
        }

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.printf("\n✅ Ultra-fast GCN completed in %.2f seconds%n", totalTime / 1000.0);
        System.out.printf("📊 Generated embeddings for %d nodes%n", embeddings.size());
    }

    // Lightweight version for very large graphs
    public void fitLightweight() {
        System.out.println("\n🚀 Starting LIGHTWEIGHT GCN Training");
        System.out.println("====================================");
        System.out.printf("Graph: %d nodes, %d edges%n", graph.getVertexCount(), graph.getEdgeCount());

        startTime = System.currentTimeMillis();

        // Even simpler approach - single propagation with smart initialization
        initializeFeaturesAndWeights();

        System.out.println("   🔄 Single propagation step...");

        // Single propagation with neighborhood aggregation
        Map<String, double[]> newEmbeddings = new HashMap<>();

        for (String node : allNodes) {
            double[] newEmb = new double[dimensions];
            double[] selfFeat = initialFeatures.get(node);
            Set<String> neighbors = graph.getNeighbors(node);
            int totalNeighbors = neighbors.size() + 1; // +1 for self

            // Self contribution
            for (int i = 0; i < dimensions; i++) {
                newEmb[i] += selfFeat[i];
            }

            // Neighbors contribution (mean aggregation)
            for (String nb : neighbors) {
                double[] nbFeat = initialFeatures.get(nb);
                for (int i = 0; i < dimensions; i++) {
                    newEmb[i] += nbFeat[i];
                }
            }

            // Normalize by degree
            for (int i = 0; i < dimensions; i++) {
                newEmb[i] /= totalNeighbors;
                // Simple non-linearity (clipped ReLU)
                newEmb[i] = Math.min(1.0, Math.max(0, newEmb[i]));
            }

            newEmbeddings.put(node, newEmb);
        }

        this.embeddings = newEmbeddings;

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.printf("\n✅ Lightweight GCN completed in %.2f seconds%n", totalTime / 1000.0);
        System.out.printf("📊 Generated embeddings for %d nodes%n", embeddings.size());
    }

    // ========= UTILS =========

    public double similarity(String vertex1, String vertex2) {
        double[] emb1 = embeddings.get(vertex1);
        double[] emb2 = embeddings.get(vertex2);

        if (emb1 == null || emb2 == null) return -1.0;

        return fastCosineSimilarity(emb1, emb2);
    }

    private double fastCosineSimilarity(double[] a, double[] b) {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < dimensions; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA == 0 || normB == 0) return 0.0;
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    public List<Map.Entry<String, Double>> mostSimilar(String vertex, int topN) {
        double[] targetEmb = embeddings.get(vertex);
        if (targetEmb == null) {
            System.out.println("❌ Target vertex not found in embeddings: " + vertex);
            return new ArrayList<>();
        }

        System.out.printf("🔍 Finding %d most similar nodes to '%s'...%n", topN, vertex);

        PriorityQueue<Map.Entry<String, Double>> heap =
                new PriorityQueue<>(topN + 1, Map.Entry.comparingByValue());

        int processed = 0;
        int total = embeddings.size() - 1;

        for (Map.Entry<String, double[]> entry : embeddings.entrySet()) {
            if (!entry.getKey().equals(vertex)) {
                double sim = fastCosineSimilarity(targetEmb, entry.getValue());

                heap.offer(new AbstractMap.SimpleEntry<>(entry.getKey(), sim));
                if (heap.size() > topN) {
                    heap.poll();
                }

                processed++;
                if (processed % 5000 == 0) {
                    System.out.printf("   📊 Processed %d/%d comparisons%n", processed, total);
                }
            }
        }

        List<Map.Entry<String, Double>> result = new ArrayList<>(heap);
        result.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));

        System.out.printf("✅ Similarity search completed for %d nodes%n", processed);
        return result;
    }

    public Map<String, double[]> getEmbeddings() {
        return new HashMap<>(embeddings);
    }

    public void saveEmbeddings(String filename) {
        System.out.printf("💾 Saving embeddings to '%s'...%n", filename);
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            int saved = 0;
            for (Map.Entry<String, double[]> entry : embeddings.entrySet()) {
                writer.print(entry.getKey());
                for (double value : entry.getValue()) {
                    writer.printf(" %.6f", value);
                }
                writer.println();
                saved++;

                if (saved % 5000 == 0) {
                    System.out.printf("   📊 Saved %d/%d embeddings%n", saved, embeddings.size());
                }
            }
            System.out.printf("✅ Saved %d embeddings to '%s'%n", saved, filename);
        } catch (IOException e) {
            System.err.println("❌ Error saving embeddings: " + e.getMessage());
        }
    }

    public void loadEmbeddings(String filename) {
        System.out.printf("📥 Loading embeddings from '%s'...%n", filename);
        embeddings.clear();
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            int loaded = 0;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split("\\s+");
                if (parts.length >= dimensions + 1) {
                    String node = parts[0];
                    double[] embedding = new double[dimensions];
                    for (int i = 0; i < dimensions; i++) {
                        embedding[i] = Double.parseDouble(parts[i + 1]);
                    }
                    embeddings.put(node, embedding);
                    loaded++;

                    if (loaded % 5000 == 0) {
                        System.out.printf("   📊 Loaded %d embeddings%n", loaded);
                    }
                }
            }
            System.out.printf("✅ Loaded %d embeddings from '%s'%n", loaded, filename);
        } catch (IOException e) {
            System.err.println("❌ Error loading embeddings: " + e.getMessage());
        }
    }

    public Graph getGraph() {
        return graph;
    }

    public int getEmbeddingDimensions() {
        return dimensions;
    }

    public int getEmbeddingCount() {
        return embeddings.size();
    }

    // Method to get embedding statistics
    public void printEmbeddingStats() {
        if (embeddings.isEmpty()) {
            System.out.println("❌ No embeddings available");
            return;
        }

        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        double sum = 0.0;
        int totalValues = 0;

        for (double[] emb : embeddings.values()) {
            for (double val : emb) {
                min = Math.min(min, val);
                max = Math.max(max, val);
                sum += val;
                totalValues++;
            }
        }

        double mean = sum / totalValues;

        // Calculate standard deviation
        double variance = 0.0;
        for (double[] emb : embeddings.values()) {
            for (double val : emb) {
                variance += Math.pow(val - mean, 2);
            }
        }
        variance /= totalValues;
        double stdDev = Math.sqrt(variance);

        System.out.println("📊 Embedding Statistics:");
        System.out.printf("   Dimensions: %d%n", dimensions);
        System.out.printf("   Node count: %d%n", embeddings.size());
        System.out.printf("   Value range: [%.4f, %.4f]%n", min, max);
        System.out.printf("   Mean: %.4f%n", mean);
        System.out.printf("   Std Dev: %.4f%n", stdDev);
    }

    // Method to validate embeddings
    public boolean validateEmbeddings() {
        if (embeddings.isEmpty()) {
            System.out.println("❌ No embeddings to validate");
            return false;
        }

        int expectedDim = dimensions;
        for (Map.Entry<String, double[]> entry : embeddings.entrySet()) {
            if (entry.getValue().length != expectedDim) {
                System.out.printf("❌ Invalid embedding dimension for node %s: expected %d, got %d%n",
                        entry.getKey(), expectedDim, entry.getValue().length);
                return false;
            }

            // Check for NaN or infinite values
            for (double val : entry.getValue()) {
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    System.out.printf("❌ Invalid value in embedding for node %s: %f%n", entry.getKey(), val);
                    return false;
                }
            }
        }

        System.out.println("✅ Embeddings validation passed");
        return true;
    }
}