package org.example.Fetch;

import java.util.*;
import java.io.*;

public class LINE {
    private final Graph graph;
    private final int dimensions;
    private final int epochs;
    private final double learningRate;
    private final int negativeSamples;

    // Two embedding tables: node and context
    private Map<String, double[]> nodeEmbeddings;     // u_i
    private Map<String, double[]> contextEmbeddings;  // u'_i
    private Random random;
    private long startTime;

    // Optimized data structures
    private List<String> allNodes;
    private List<Edge> edges;
    private int nodeCount;
    private double initialLearningRate;

    public LINE(Graph graph, int dimensions, double learningRate, int epochs, int negativeSamples) {
        this.graph = graph;
        this.dimensions = dimensions;
        this.learningRate = learningRate;
        this.initialLearningRate = learningRate;
        this.epochs = epochs;
        this.negativeSamples = negativeSamples;
        this.random = new Random(42);
        this.nodeEmbeddings = new HashMap<>();
        this.contextEmbeddings = new HashMap<>();

        // Precompute optimized structures
        this.allNodes = new ArrayList<>(graph.getAdjacencyList().keySet());
        this.nodeCount = allNodes.size();
        precomputeEdges();
    }

    private void precomputeEdges() {
        edges = new ArrayList<>();
        Set<String> processed = new HashSet<>();

        for (String u : graph.getAdjacencyList().keySet()) {
            for (String v : graph.getNeighbors(u)) {
                String edgeKey = u.compareTo(v) < 0 ? u + "|" + v : v + "|" + u;
                if (!processed.contains(edgeKey)) {
                    edges.add(new Edge(u, v, 1.0));
                    processed.add(edgeKey);
                }
            }
        }

        System.out.printf("   Precomputed %d unique edges%n", edges.size());
    }

    public void fit() {
        System.out.println("\n🚀 Starting OPTIMIZED LINE Training");
        System.out.println("===================================");
        System.out.printf("Graph: %d nodes, %d edges%n", graph.getVertexCount(), graph.getEdgeCount());
        System.out.printf("Parameters: dim=%d, lr=%.4f, epochs=%d, negSamples=%d%n",
                dimensions, learningRate, epochs, negativeSamples);

        startTime = System.currentTimeMillis();

        // Phase 1: Initialize embeddings
        System.out.println("\n🎯 Phase 1: Initializing embeddings...");
        initializeEmbeddingsOptimized();

        // Phase 2: Training with optimizations
        System.out.println("\n🎯 Phase 2: Training with optimized sampling...");
        trainOptimized();

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.printf("\n✅ LINE training completed in %.2f seconds%n", totalTime / 1000.0);
    }

    private void initializeEmbeddingsOptimized() {
        // Xavier initialization for better convergence
        double xavier = Math.sqrt(6.0 / (dimensions + 1));

        for (String vertex : allNodes) {
            // Node embedding
            double[] nodeEmb = new double[dimensions];
            // Context embedding
            double[] contextEmb = new double[dimensions];

            for (int i = 0; i < dimensions; i++) {
                nodeEmb[i] = (random.nextDouble() - 0.5) * 2.0 * xavier;
                contextEmb[i] = (random.nextDouble() - 0.5) * 2.0 * xavier;
            }

            nodeEmbeddings.put(vertex, nodeEmb);
            contextEmbeddings.put(vertex, contextEmb);
        }

        System.out.printf("      ✅ Embeddings initialized for %d nodes%n", allNodes.size());
    }

    private void trainOptimized() {
        if (edges.isEmpty()) {
            System.err.println("   ❌ No edges found! Cannot train LINE.");
            return;
        }

        // Precompute negative sampling distribution (degree-based)
        double[] samplingDistribution = computeSamplingDistribution();

        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.printf("\n   📚 Epoch %d/%d%n", epoch + 1, epochs);
            long epochStartTime = System.currentTimeMillis();
            double totalLoss = 0.0;
            int processed = 0;

            // Learning rate decay
            double currentLR = initialLearningRate * (1.0 - (double) epoch / epochs);

            // Shuffle edges in batches for better cache performance
            Collections.shuffle(edges, random);

            int batchSize = Math.min(1000, edges.size());
            List<Edge> batch = new ArrayList<>();

            for (Edge edge : edges) {
                batch.add(edge);

                if (batch.size() >= batchSize) {
                    totalLoss += processBatch(batch, currentLR, samplingDistribution);
                    processed += batch.size();
                    batch.clear();

                    // Progress reporting
                    if (processed % Math.max(1, edges.size() / 5) == 0) {
                        double progress = (double) processed / edges.size() * 100;
                        System.out.printf("      📊 Processed %d/%d edges (%.1f%%) - Avg Loss: %.6f%n",
                                processed, edges.size(), progress, totalLoss / processed);
                    }
                }
            }

            // Process remaining edges
            if (!batch.isEmpty()) {
                totalLoss += processBatch(batch, currentLR, samplingDistribution);
                processed += batch.size();
            }

            long epochTime = System.currentTimeMillis() - epochStartTime;
            double avgLoss = totalLoss / Math.max(1, processed);
            System.out.printf("      ✅ Epoch completed in %.2f seconds - Avg Loss: %.6f%n",
                    epochTime / 1000.0, avgLoss);
        }
    }

    private double[] computeSamplingDistribution() {
        // Power law distribution for negative sampling (like word2vec)
        double[] distribution = new double[nodeCount];
        double sum = 0.0;

        for (int i = 0; i < nodeCount; i++) {
            String node = allNodes.get(i);
            int degree = graph.getNeighbors(node).size();
            // Use 3/4 power like in original word2vec
            distribution[i] = Math.pow(degree, 0.75);
            sum += distribution[i];
        }

        // Normalize
        for (int i = 0; i < nodeCount; i++) {
            distribution[i] /= sum;
        }

        return distribution;
    }

    private double processBatch(List<Edge> batch, double currentLR, double[] samplingDistribution) {
        double batchLoss = 0.0;

        for (Edge edge : batch) {
            String u = edge.u;
            String v = edge.v;

            double[] uEmb = nodeEmbeddings.get(u);
            double[] vContextEmb = contextEmbeddings.get(v);

            if (uEmb == null || vContextEmb == null) continue;

            // Positive update
            double posLoss = updatePairOptimized(uEmb, vContextEmb, true, currentLR);
            batchLoss += posLoss;

            // Negative sampling with optimized distribution
            for (int ns = 0; ns < negativeSamples; ns++) {
                int negIndex = sampleNegativeNode(samplingDistribution);
                String negNode = allNodes.get(negIndex);

                // Skip if negative sample is actually a neighbor (optional)
                if (!graph.getNeighbors(u).contains(negNode)) {
                    double[] negContextEmb = contextEmbeddings.get(negNode);
                    if (negContextEmb != null) {
                        batchLoss += updatePairOptimized(uEmb, negContextEmb, false, currentLR);
                    }
                }
            }
        }

        return batchLoss;
    }

    private int sampleNegativeNode(double[] distribution) {
        // Alias method for efficient sampling (simplified version)
        double rand = random.nextDouble();
        double cumulative = 0.0;

        for (int i = 0; i < distribution.length; i++) {
            cumulative += distribution[i];
            if (rand <= cumulative) {
                return i;
            }
        }

        return random.nextInt(nodeCount); // fallback
    }

    private double updatePairOptimized(double[] nodeEmb, double[] contextEmb, boolean positive, double currentLR) {
        // Fast dot product with bounds checking
        double dot = fastDotProduct(nodeEmb, contextEmb);

        // Clipped sigmoid for numerical stability
        double sigmoid = 1.0 / (1.0 + Math.exp(-Math.max(-15.0, Math.min(15.0, dot))));
        double label = positive ? 1.0 : 0.0;
        double loss = - (label * Math.log(sigmoid + 1e-12) + (1 - label) * Math.log(1 - sigmoid + 1e-12));

        double gradient = (sigmoid - label) * currentLR;

        // Update embeddings with momentum-like effect
        for (int i = 0; i < dimensions; i++) {
            double nodeGrad = gradient * contextEmb[i];
            double contextGrad = gradient * nodeEmb[i];

            // Apply updates
            nodeEmb[i] -= nodeGrad;
            contextEmb[i] -= contextGrad;
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

    // Ultra-fast version for large graphs
    public void fitFast() {
        System.out.println("\n🚀 Starting ULTRA-FAST LINE Training");
        System.out.println("====================================");
        System.out.printf("Graph: %d nodes, %d edges%n", graph.getVertexCount(), graph.getEdgeCount());

        startTime = System.currentTimeMillis();

        // Simple initialization
        initializeEmbeddingsOptimized();

        // Reduced parameters for speed
        int fastEpochs = Math.min(3, epochs);
        int fastNegativeSamples = Math.min(2, negativeSamples);

        System.out.printf("Fast parameters: epochs=%d, negSamples=%d%n", fastEpochs, fastNegativeSamples);

        for (int epoch = 0; epoch < fastEpochs; epoch++) {
            System.out.printf("   🔄 Fast epoch %d/%d%n", epoch + 1, fastEpochs);
            long epochStart = System.currentTimeMillis();

            // Simple sampling without complex distributions
            Collections.shuffle(edges, random);

            int samples = Math.min(5000, edges.size()); // Limit samples for speed
            double epochLoss = 0.0;

            for (int i = 0; i < samples; i++) {
                Edge edge = edges.get(i);
                String u = edge.u;
                String v = edge.v;

                double[] uEmb = nodeEmbeddings.get(u);
                double[] vContextEmb = contextEmbeddings.get(v);

                if (uEmb != null && vContextEmb != null) {
                    // Positive update only
                    epochLoss += updatePairOptimized(uEmb, vContextEmb, true, learningRate * 0.5);

                    // One negative sample
                    String negNode = allNodes.get(random.nextInt(nodeCount));
                    double[] negContextEmb = contextEmbeddings.get(negNode);
                    if (negContextEmb != null) {
                        epochLoss += updatePairOptimized(uEmb, negContextEmb, false, learningRate * 0.5);
                    }
                }

                if ((i + 1) % 1000 == 0) {
                    System.out.printf("      📊 Processed %d/%d samples%n", i + 1, samples);
                }
            }

            long epochTime = System.currentTimeMillis() - epochStart;
            System.out.printf("      ✅ Epoch completed in %.2f seconds - Avg Loss: %.6f%n",
                    epochTime / 1000.0, epochLoss / samples);
        }

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.printf("\n✅ Ultra-fast LINE completed in %.2f seconds%n", totalTime / 1000.0);
        System.out.printf("📊 Generated embeddings for %d nodes%n", nodeEmbeddings.size());
    }

    // ========= UTILS =========

    public double similarity(String vertex1, String vertex2) {
        double[] emb1 = nodeEmbeddings.get(vertex1);
        double[] emb2 = nodeEmbeddings.get(vertex2);

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
        double[] targetEmb = nodeEmbeddings.get(vertex);
        if (targetEmb == null) {
            System.out.println("❌ Target vertex not found in embeddings: " + vertex);
            return new ArrayList<>();
        }

        System.out.printf("🔍 Finding %d most similar nodes to '%s'...%n", topN, vertex);

        PriorityQueue<Map.Entry<String, Double>> heap =
                new PriorityQueue<>(topN + 1, Map.Entry.comparingByValue());

        int processed = 0;
        int total = nodeEmbeddings.size() - 1;

        for (Map.Entry<String, double[]> entry : nodeEmbeddings.entrySet()) {
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
        return new HashMap<>(nodeEmbeddings);
    }

    public void saveEmbeddings(String filename) {
        System.out.printf("💾 Saving embeddings to '%s'...%n", filename);
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            int saved = 0;
            for (Map.Entry<String, double[]> entry : nodeEmbeddings.entrySet()) {
                writer.print(entry.getKey());
                for (double value : entry.getValue()) {
                    writer.printf(" %.6f", value);
                }
                writer.println();
                saved++;

                if (saved % 5000 == 0) {
                    System.out.printf("   📊 Saved %d/%d embeddings%n", saved, nodeEmbeddings.size());
                }
            }
            System.out.printf("✅ Saved %d embeddings to '%s'%n", saved, filename);
        } catch (IOException e) {
            System.err.println("❌ Error saving embeddings: " + e.getMessage());
        }
    }

    public void loadEmbeddings(String filename) {
        System.out.printf("📥 Loading embeddings from '%s'...%n", filename);
        nodeEmbeddings.clear();
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
                    nodeEmbeddings.put(node, embedding);
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
        return nodeEmbeddings.size();
    }

    // Method to get embedding statistics
    public void printEmbeddingStats() {
        if (nodeEmbeddings.isEmpty()) {
            System.out.println("❌ No embeddings available");
            return;
        }

        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        double sum = 0.0;
        int totalValues = 0;

        for (double[] emb : nodeEmbeddings.values()) {
            for (double val : emb) {
                min = Math.min(min, val);
                max = Math.max(max, val);
                sum += val;
                totalValues++;
            }
        }

        double mean = sum / totalValues;

        System.out.println("📊 LINE Embedding Statistics:");
        System.out.printf("   Dimensions: %d%n", dimensions);
        System.out.printf("   Node count: %d%n", nodeEmbeddings.size());
        System.out.printf("   Value range: [%.4f, %.4f]%n", min, max);
        System.out.printf("   Mean: %.4f%n", mean);
    }

    // Helper class for edges
    private static class Edge {
        String u, v;
        double weight;

        Edge(String u, String v, double weight) {
            this.u = u;
            this.v = v;
            this.weight = weight;
        }
    }
}