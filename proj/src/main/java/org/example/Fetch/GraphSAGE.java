package org.example.Fetch;

import java.util.*;
import java.io.*;

public class GraphSAGE {
    private final Graph graph;
    private final int dimensions;
    private final int numSamples;
    private final int epochs;
    private final double learningRate;

    private Map<String, double[]> embeddings;
    private Random random;
    private long startTime;

    // Optimized data structures
    private List<String> allNodes;
    private int nodeCount;
    private double initialLearningRate;

    public GraphSAGE(Graph graph, int dimensions, int numSamples, double learningRate, int epochs) {
        this.graph = graph;
        this.dimensions = dimensions;
        this.numSamples = numSamples;
        this.learningRate = learningRate;
        this.initialLearningRate = learningRate;
        this.epochs = epochs;
        this.random = new Random(42);
        this.embeddings = new HashMap<>();

        // Precompute optimized structures
        this.allNodes = new ArrayList<>(graph.getAdjacencyList().keySet());
        this.nodeCount = allNodes.size();
    }

    public void fit() {
        System.out.println("\n🚀 Starting OPTIMIZED GraphSAGE Training");
        System.out.println("========================================");
        System.out.printf("Graph: %d nodes, %d edges%n", graph.getVertexCount(), graph.getEdgeCount());
        System.out.printf("Parameters: dim=%d, numSamples=%d, lr=%.4f, epochs=%d%n",
                dimensions, numSamples, learningRate, epochs);

        startTime = System.currentTimeMillis();

        // Phase 1: Initialize embeddings
        System.out.println("\n🎯 Phase 1: Initializing embeddings...");
        initializeEmbeddingsOptimized();

        // Phase 2: Training with optimizations
        System.out.println("\n🎯 Phase 2: Training with optimized sampling...");
        trainOptimized();

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.printf("\n✅ GraphSAGE training completed in %.2f seconds%n", totalTime / 1000.0);
    }

    private void initializeEmbeddingsOptimized() {
        // Xavier initialization for better convergence
        double xavier = Math.sqrt(6.0 / (dimensions + 1));

        for (String vertex : allNodes) {
            double[] embedding = new double[dimensions];
            for (int i = 0; i < dimensions; i++) {
                embedding[i] = (random.nextDouble() - 0.5) * 2.0 * xavier;
            }
            embeddings.put(vertex, embedding);
        }

        System.out.printf("      ✅ Embeddings initialized for %d nodes%n", allNodes.size());
    }

    private void trainOptimized() {
        // Precompute negative sampling distribution
        double[] samplingDistribution = computeSamplingDistribution();

        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.printf("\n   📚 Epoch %d/%d%n", epoch + 1, epochs);
            long epochStartTime = System.currentTimeMillis();
            double totalLoss = 0.0;
            int processed = 0;

            // Learning rate decay
            double currentLR = initialLearningRate * (1.0 - (double) epoch / epochs);

            // Shuffle nodes in batches
            List<String> shuffledNodes = new ArrayList<>(allNodes);
            Collections.shuffle(shuffledNodes, random);

            int batchSize = Math.min(500, nodeCount);
            List<String> batch = new ArrayList<>();

            for (String node : shuffledNodes) {
                batch.add(node);

                if (batch.size() >= batchSize) {
                    totalLoss += processBatch(batch, currentLR, samplingDistribution);
                    processed += batch.size();
                    batch.clear();

                    // Progress reporting
                    if (processed % Math.max(1, nodeCount / 5) == 0) {
                        double progress = (double) processed / nodeCount * 100;
                        System.out.printf("      📊 Processed %d/%d nodes (%.1f%%) - Avg Loss: %.6f%n",
                                processed, nodeCount, progress, totalLoss / processed);
                    }
                }
            }

            // Process remaining nodes
            if (!batch.isEmpty()) {
                totalLoss += processBatch(batch, currentLR, samplingDistribution);
                processed += batch.size();
            }

            long epochTime = System.currentTimeMillis() - epochStartTime;
            double avgLoss = totalLoss / Math.max(1, processed);
            System.out.printf("      ✅ Epoch completed in %.2f seconds - Avg Loss: %.6f%n",
                    epochTime / 1000.0, avgLoss);
        }

        // Final aggregation without sampling
        computeFinalEmbeddings();
    }

    private double[] computeSamplingDistribution() {
        double[] distribution = new double[nodeCount];
        double sum = 0.0;

        for (int i = 0; i < nodeCount; i++) {
            String node = allNodes.get(i);
            int degree = graph.getNeighbors(node).size();
            distribution[i] = Math.pow(degree + 1, 0.75); // +1 to avoid 0
            sum += distribution[i];
        }

        // Normalize
        for (int i = 0; i < nodeCount; i++) {
            distribution[i] /= sum;
        }

        return distribution;
    }

    private double processBatch(List<String> batch, double currentLR, double[] samplingDistribution) {
        double batchLoss = 0.0;

        for (String node : batch) {
            Set<String> neighbors = graph.getNeighbors(node);
            if (neighbors.isEmpty()) continue;

            double[] selfEmb = embeddings.get(node);

            // Sample neighbors
            List<String> sampledNeighbors = sampleNeighbors(neighbors);
            double[] neighborAgg = aggregateNeighbors(sampledNeighbors);

            // Positive update
            double posLoss = updatePairOptimized(selfEmb, neighborAgg, true, currentLR);
            batchLoss += posLoss;

            // Negative sampling
            for (int ns = 0; ns < 2; ns++) { // 2 negative samples
                int negIndex = sampleNegativeNode(samplingDistribution);
                String negNode = allNodes.get(negIndex);

                if (!neighbors.contains(negNode) && !negNode.equals(node)) {
                    double[] negEmb = embeddings.get(negNode);
                    if (negEmb != null) {
                        batchLoss += updatePairOptimized(selfEmb, negEmb, false, currentLR);
                    }
                }
            }
        }

        return batchLoss;
    }

    private List<String> sampleNeighbors(Set<String> neighbors) {
        List<String> neighborList = new ArrayList<>(neighbors);
        if (neighborList.size() <= numSamples) {
            return neighborList;
        }

        Collections.shuffle(neighborList, random);
        return neighborList.subList(0, numSamples);
    }

    private double[] aggregateNeighbors(List<String> neighbors) {
        double[] aggregation = new double[dimensions];

        for (String neighbor : neighbors) {
            double[] nbEmb = embeddings.get(neighbor);
            if (nbEmb != null) {
                for (int i = 0; i < dimensions; i++) {
                    aggregation[i] += nbEmb[i];
                }
            }
        }

        // Normalize
        if (!neighbors.isEmpty()) {
            for (int i = 0; i < dimensions; i++) {
                aggregation[i] /= neighbors.size();
            }
        }

        return aggregation;
    }

    private int sampleNegativeNode(double[] distribution) {
        double rand = random.nextDouble();
        double cumulative = 0.0;

        for (int i = 0; i < distribution.length; i++) {
            cumulative += distribution[i];
            if (rand <= cumulative) {
                return i;
            }
        }

        return random.nextInt(nodeCount);
    }

    private double updatePairOptimized(double[] selfEmb, double[] otherEmb, boolean positive, double currentLR) {
        double dot = fastDotProduct(selfEmb, otherEmb);

        double sigmoid = 1.0 / (1.0 + Math.exp(-Math.max(-15.0, Math.min(15.0, dot))));
        double label = positive ? 1.0 : 0.0;
        double loss = - (label * Math.log(sigmoid + 1e-12) + (1 - label) * Math.log(1 - sigmoid + 1e-12));

        double gradient = (sigmoid - label) * currentLR;

        // Update embeddings
        for (int i = 0; i < dimensions; i++) {
            double selfGrad = gradient * otherEmb[i];
            double otherGrad = gradient * selfEmb[i];

            selfEmb[i] -= selfGrad;
            if (positive) { // Only update neighbor embeddings for positive pairs
                otherEmb[i] -= otherGrad;
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

    private void computeFinalEmbeddings() {
        System.out.println("\n   🔁 Computing final embeddings with full aggregation...");
        Map<String, double[]> finalEmbeddings = new HashMap<>();

        for (String node : allNodes) {
            Set<String> neighbors = graph.getNeighbors(node);
            double[] selfEmb = embeddings.get(node).clone();

            if (!neighbors.isEmpty()) {
                double[] neighborAgg = aggregateNeighbors(new ArrayList<>(neighbors));

                // Combine self and neighbor embeddings
                for (int i = 0; i < dimensions; i++) {
                    selfEmb[i] = Math.max(0, (selfEmb[i] + neighborAgg[i]) / 2.0);
                }
            }

            finalEmbeddings.put(node, selfEmb);
        }

        this.embeddings = finalEmbeddings;
        System.out.println("   ✅ Final embeddings computed");
    }

    // Ultra-fast version
    public void fitFast() {
        System.out.println("\n🚀 Starting ULTRA-FAST GraphSAGE Training");
        System.out.println("==========================================");
        System.out.printf("Graph: %d nodes, %d edges%n", graph.getVertexCount(), graph.getEdgeCount());

        startTime = System.currentTimeMillis();

        // Simple initialization
        initializeEmbeddingsOptimized();

        // Reduced parameters for speed
        int fastEpochs = Math.min(2, epochs);
        int fastSamples = Math.min(5, numSamples);

        System.out.printf("Fast parameters: epochs=%d, samples=%d%n", fastEpochs, fastSamples);

        for (int epoch = 0; epoch < fastEpochs; epoch++) {
            System.out.printf("   🔄 Fast epoch %d/%d%n", epoch + 1, fastEpochs);
            long epochStart = System.currentTimeMillis();

            // Simple training with limited samples
            List<String> shuffledNodes = new ArrayList<>(allNodes);
            Collections.shuffle(shuffledNodes, random);

            int samples = Math.min(2000, nodeCount);
            double epochLoss = 0.0;

            for (int i = 0; i < samples; i++) {
                String node = shuffledNodes.get(i);
                Set<String> neighbors = graph.getNeighbors(node);
                if (neighbors.isEmpty()) continue;

                double[] selfEmb = embeddings.get(node);
                List<String> sampledNeighbors = sampleNeighbors(neighbors);
                double[] neighborAgg = aggregateNeighbors(sampledNeighbors);

                // Positive update
                epochLoss += updatePairOptimized(selfEmb, neighborAgg, true, learningRate * 0.5);

                // One negative sample
                String negNode = allNodes.get(random.nextInt(nodeCount));
                if (!neighbors.contains(negNode)) {
                    double[] negEmb = embeddings.get(negNode);
                    if (negEmb != null) {
                        epochLoss += updatePairOptimized(selfEmb, negEmb, false, learningRate * 0.5);
                    }
                }

                if ((i + 1) % 500 == 0) {
                    System.out.printf("      📊 Processed %d/%d samples%n", i + 1, samples);
                }
            }

            long epochTime = System.currentTimeMillis() - epochStart;
            System.out.printf("      ✅ Epoch completed in %.2f seconds - Avg Loss: %.6f%n",
                    epochTime / 1000.0, epochLoss / samples);
        }

        // Quick final aggregation
        computeFinalEmbeddings();

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.printf("\n✅ Ultra-fast GraphSAGE completed in %.2f seconds%n", totalTime / 1000.0);
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

    public Graph getGraph() {
        return graph;
    }

    public int getEmbeddingDimensions() {
        return dimensions;
    }

    public int getEmbeddingCount() {
        return embeddings.size();
    }

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

        System.out.println("📊 GraphSAGE Embedding Statistics:");
        System.out.printf("   Dimensions: %d%n", dimensions);
        System.out.printf("   Node count: %d%n", embeddings.size());
        System.out.printf("   Value range: [%.4f, %.4f]%n", min, max);
        System.out.printf("   Mean: %.4f%n", mean);
    }
}