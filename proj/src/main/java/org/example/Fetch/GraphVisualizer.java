package org.example.Fetch;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.List;

public class GraphVisualizer extends JFrame {
    private Map<String, double[]> embeddings;
    private Map<String, Point2D> coordinates;
    private Graph graph;
    private JTextArea infoArea;
    private GraphPanel graphPanel;

    public GraphVisualizer(Map<String, double[]> embeddings, Graph graph) {
        this.embeddings = embeddings;
        this.graph = graph;
        this.coordinates = reduceTo2D(); // Convert 64D → 2D for visualization

        initializeUI();
        showEmbeddingInfo();
    }

    // Convert 64-dimensional vectors to 2D coordinates using PCA-like approach
    private Map<String, Point2D> reduceTo2D() {
        Map<String, Point2D> coords = new HashMap<>();

        System.out.println("🔄 Reducing 64D embeddings to 2D for visualization...");

        for (Map.Entry<String, double[]> entry : embeddings.entrySet()) {
            String node = entry.getKey();
            double[] embedding = entry.getValue();

            // Simple projection: use first two principal components
            // In practice, you'd use proper PCA, but this works for visualization
            double x = 0, y = 0;

            // Use different weights for different dimensions to spread points
            for (int i = 0; i < embedding.length; i++) {
                double weight = Math.sin(i * 0.1) + 1.0; // Varying weights
                if (i % 2 == 0) {
                    x += embedding[i] * weight;
                } else {
                    y += embedding[i] * weight;
                }
            }

            // Scale and center
            x = 400 + (x * 300);
            y = 300 + (y * 300);

            coords.put(node, new Point2D(x, y));
        }

        System.out.println("✅ Reduced " + coords.size() + " nodes to 2D coordinates");
        return coords;
    }

    private void showEmbeddingInfo() {
        if (!embeddings.isEmpty()) {
            double[] sampleVector = embeddings.values().iterator().next();
            System.out.println("📐 Each vector has " + sampleVector.length + " dimensions");
            System.out.println("🎯 Displaying in 2D using dimensionality reduction");
        }
    }

    private void initializeUI() {
        setTitle("Graph Visualization - 64D Node2Vec Embeddings (Reduced to 2D)");
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setSize(1200, 800);
        setLayout(new BorderLayout());

        // Visualization panel
        graphPanel = new GraphPanel();
        graphPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                handleMouseClick(e.getX(), e.getY());
            }
        });
        add(graphPanel, BorderLayout.CENTER);

        // Information panel
        infoArea = new JTextArea(15, 40);
        infoArea.setEditable(false);
        infoArea.setFont(new Font("Monospaced", Font.PLAIN, 12));
        JScrollPane scrollPane = new JScrollPane(infoArea);
        add(scrollPane, BorderLayout.EAST);

        // Control panel
        JPanel controlPanel = createControlPanel();
        add(controlPanel, BorderLayout.SOUTH);

        setVisible(true);
    }

    private JPanel createControlPanel() {
        JPanel panel = new JPanel();

        JButton similarityBtn = new JButton("Calculate Cosine Similarity");
        similarityBtn.addActionListener(e -> calculateSimilarity());

        JButton showCoordsBtn = new JButton("Show 2D Coordinates");
        showCoordsBtn.addActionListener(e -> showAllCoordinates());

        JButton showVectorBtn = new JButton("Show Full Vector");
        showVectorBtn.addActionListener(e -> showFullVector());

        JButton findSimilarBtn = new JButton("Find Most Similar");
        findSimilarBtn.addActionListener(e -> findSimilarNodes());

        JButton clearBtn = new JButton("Clear");
        clearBtn.addActionListener(e -> infoArea.setText(""));

        panel.add(similarityBtn);
        panel.add(showCoordsBtn);
        panel.add(showVectorBtn);
        panel.add(findSimilarBtn);
        panel.add(clearBtn);

        return panel;
    }

    private void handleMouseClick(int x, int y) {
        // Find clicked node
        for (Map.Entry<String, Point2D> entry : coordinates.entrySet()) {
            Point2D point = entry.getValue();
            double distance = Math.sqrt(Math.pow(x - point.getX(), 2) + Math.pow(y - point.getY(), 2));
            if (distance < 10) {
                showNodeInfo(entry.getKey());
                break;
            }
        }
    }

    private void showNodeInfo(String node) {
        Point2D point = coordinates.get(node);
        String type = graph.getAuthorVertices().contains(node) ? "Author" : "Publication";
        double[] vector = embeddings.get(node);

        infoArea.append("\n🎯 Clicked Node: " + node + "\n");
        infoArea.append("   Type: " + type + "\n");
        infoArea.append(String.format("   2D Coordinates: (%.2f, %.2f)\n", point.getX(), point.getY()));
        infoArea.append("   Original dimensions: " + vector.length + "\n");
        infoArea.append("   Degree: " + graph.getDegree(node) + "\n");

        // Show first few vector values
        infoArea.append("   First 5 coordinates: ");
        for (int i = 0; i < Math.min(5, vector.length); i++) {
            infoArea.append(String.format("%.4f ", vector[i]));
        }
        infoArea.append("\n");
    }

    private void calculateSimilarity() {
        String node1 = JOptionPane.showInputDialog(this, "Enter first node:");
        String node2 = JOptionPane.showInputDialog(this, "Enter second node:");

        if (node1 != null && node2 != null && embeddings.containsKey(node1) && embeddings.containsKey(node2)) {
            double similarity = cosineSimilarity(embeddings.get(node1), embeddings.get(node2));
            infoArea.append(String.format("\n🔍 Cosine Similarity (64D vectors):\n"));
            infoArea.append(String.format("   '%s' ↔ '%s': %.4f\n", node1, node2, similarity));

            // Interpretation
            if (similarity > 0.8) {
                infoArea.append("   📈 Very similar in embedding space\n");
            } else if (similarity > 0.5) {
                infoArea.append("   📊 Moderately similar\n");
            } else if (similarity > 0.2) {
                infoArea.append("   📉 Slightly similar\n");
            } else if (similarity > -0.2) {
                infoArea.append("   📋 Not similar\n");
            } else {
                infoArea.append("   ⚠️  Dissimilar (negative correlation)\n");
            }
        } else {
            infoArea.append("\n❌ Error: One or both nodes not found\n");
        }
    }

    private void showAllCoordinates() {
        infoArea.setText("=== 2D COORDINATES (from 64D reduction) ===\n");
        int count = 0;
        for (Map.Entry<String, Point2D> entry : coordinates.entrySet()) {
            if (count++ > 50) { // Show first 50 to avoid overflow
                infoArea.append("... and " + (coordinates.size() - 50) + " more nodes\n");
                break;
            }
            Point2D point = entry.getValue();
            String type = graph.getAuthorVertices().contains(entry.getKey()) ? "A" : "P";
            infoArea.append(String.format("%s [%s]: (%.1f, %.1f)\n",
                    entry.getKey(), type, point.getX(), point.getY()));
        }
    }

    private void showFullVector() {
        String node = JOptionPane.showInputDialog(this, "Enter node to show full vector:");
        if (node != null && embeddings.containsKey(node)) {
            double[] vector = embeddings.get(node);
            infoArea.append("\n📐 Full 64D Vector for: " + node + "\n");

            for (int i = 0; i < vector.length; i++) {
                infoArea.append(String.format("%8.4f", vector[i]));
                if ((i + 1) % 8 == 0) infoArea.append("\n"); // New line every 8 values
            }
            infoArea.append("\n");
        }
    }

    private void findSimilarNodes() {
        String node = JOptionPane.showInputDialog(this, "Enter node to find similar nodes:");
        if (node != null && embeddings.containsKey(node)) {
            infoArea.append("\n🔍 Most Similar Nodes to: " + node + "\n");
            infoArea.append("   (based on 64D cosine similarity)\n");

            List<Map.Entry<String, Double>> similar = mostSimilar(node, 10);
            for (int i = 0; i < similar.size(); i++) {
                Map.Entry<String, Double> entry = similar.get(i);
                String similarNode = entry.getKey();
                String type = graph.getAuthorVertices().contains(similarNode) ? "Author" : "Publication";
                infoArea.append(String.format("%2d. %-40s [%s]: %.4f\n",
                        i + 1, similarNode, type, entry.getValue()));
            }
        }
    }

    private double cosineSimilarity(double[] vectorA, double[] vectorB) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    private List<Map.Entry<String, Double>> mostSimilar(String vertex, int topN) {
        List<Map.Entry<String, Double>> similarities = new ArrayList<>();

        for (Map.Entry<String, double[]> entry : embeddings.entrySet()) {
            if (!entry.getKey().equals(vertex)) {
                double sim = cosineSimilarity(embeddings.get(vertex), entry.getValue());
                similarities.add(new AbstractMap.SimpleEntry<>(entry.getKey(), sim));
            }
        }

        similarities.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        return similarities.subList(0, Math.min(topN, similarities.size()));
    }

    // Inner class for graph drawing
    class GraphPanel extends JPanel {
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2d = (Graphics2D) g;

            // Draw edges
            g2d.setColor(Color.LIGHT_GRAY);
            for (String node1 : coordinates.keySet()) {
                Point2D p1 = coordinates.get(node1);
                for (String neighbor : graph.getNeighbors(node1)) {
                    if (coordinates.containsKey(neighbor)) {
                        Point2D p2 = coordinates.get(neighbor);
                        g2d.drawLine((int)p1.getX(), (int)p1.getY(),
                                (int)p2.getX(), (int)p2.getY());
                    }
                }
            }

            // Draw nodes
            for (Map.Entry<String, Point2D> entry : coordinates.entrySet()) {
                String node = entry.getKey();
                Point2D point = entry.getValue();

                // Color authors and publications differently
                if (graph.getAuthorVertices().contains(node)) {
                    g2d.setColor(Color.BLUE);
                    g2d.fillOval((int)point.getX() - 6, (int)point.getY() - 6, 12, 12);
                } else {
                    g2d.setColor(Color.RED);
                    g2d.fillRect((int)point.getX() - 5, (int)point.getY() - 5, 10, 10);
                }

                // Draw node label with coordinates
                g2d.setColor(Color.BLACK);
                String shortName = node.length() > 20 ? node.substring(0, 20) + "..." : node;
                g2d.drawString(shortName, (int)point.getX() + 15, (int)point.getY());
                g2d.setColor(Color.DARK_GRAY);
                g2d.drawString(String.format("(%.0f,%.0f)", point.getX(), point.getY()),
                        (int)point.getX() + 15, (int)point.getY() + 15);
            }

            // Draw legend
            g2d.setColor(Color.BLUE);
            g2d.fillOval(20, 20, 10, 10);
            g2d.setColor(Color.BLACK);
            g2d.drawString("Authors", 35, 28);

            g2d.setColor(Color.RED);
            g2d.fillRect(20, 40, 10, 10);
            g2d.setColor(Color.BLACK);
            g2d.drawString("Publications", 35, 48);

            g2d.setColor(Color.DARK_GRAY);
            g2d.drawString("64D Node2Vec embeddings reduced to 2D", 20, 70);
        }
    }

    // Simple 2D point class
    static class Point2D {
        private final double x, y;

        public Point2D(double x, double y) {
            this.x = x;
            this.y = y;
        }

        public double getX() { return x; }
        public double getY() { return y; }
    }
}