package org.example;

import javafx.animation.FadeTransition;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.effect.DropShadow;
import javafx.scene.effect.Glow;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.*;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import javafx.scene.text.TextAlignment;
import javafx.stage.Stage;
import javafx.util.Duration;
import org.example.Fetch.BipartiteGraphBuilder;
import org.example.Fetch.Graph;
import org.example.Fetch.Node2Vec;
import org.example.Fetch.DeepWalk;
import org.example.Fetch.GCN;
import org.example.Fetch.LINE;
import org.example.Fetch.GraphSAGE;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class Main extends Application {

    private Graph graph;
    private Group graphGroup;
    private Pane graphPane;
    private Pane node2VecPane;
    private Pane deepWalkPane;
    private Pane gcnPane;
    private Pane linePane;
    private Pane graphSagePane;
    private Map<String, Circle> nodeCircles = new ConcurrentHashMap<>();
    private Map<String, Text> nodeLabels = new ConcurrentHashMap<>();
    private boolean edgesDrawn = false;
    private double scale = 1.0;
    private Node2Vec node2vec;
    private DeepWalk deepwalk;
    private GCN gcn;
    private LINE line;
    private GraphSAGE graphSage;
    private Map<String, double[]> node2vecEmbeddings;
    private Map<String, double[]> deepwalkEmbeddings;
    private Map<String, double[]> gcnEmbeddings;
    private Map<String, double[]> lineEmbeddings;
    private Map<String, double[]> graphSageEmbeddings;
    private boolean node2vecCalculated = false;
    private boolean deepwalkCalculated = false;
    private boolean gcnCalculated = false;
    private boolean lineCalculated = false;
    private boolean graphSageCalculated = false;

    private static final int MAX_DISPLAYED_NODES = 12;
    private static final int MAX_DISPLAYED_EDGES = 20;

    // Colors
    private final Color AUTHOR_COLOR = Color.rgb(65, 105, 225, 0.9);
    private final Color PUBLICATION_COLOR = Color.rgb(220, 20, 60, 0.9);
    private final Color AUTHOR_STROKE = Color.rgb(30, 144, 255);
    private final Color PUBLICATION_STROKE = Color.rgb(255, 69, 0);
    private final Color AUTHOR_LABEL = Color.rgb(173, 216, 230);
    private final Color PUBLICATION_LABEL = Color.rgb(255, 182, 193);
    private final Color EDGE_COLOR = Color.rgb(169, 169, 169, 0.6);
    private final Color DASHED_INDICATOR_COLOR = Color.rgb(150, 150, 150, 0.4);
    private final Color PREDICTION_COLOR = Color.rgb(46, 204, 113, 0.6);
    private final Color DEEPWALK_COLOR = Color.rgb(155, 89, 182, 0.7);
    private final Color NODE2VEC_COLOR = Color.rgb(52, 152, 219, 0.7);
    private final Color GCN_COLOR = Color.rgb(231, 76, 60, 0.7);
    private final Color LINE_COLOR = Color.rgb(241, 196, 15, 0.7);
    private final Color GRAPHSAGE_COLOR = Color.rgb(39, 174, 96, 0.7);

    private List<String> cachedTopAuthors = null;
    private List<String> cachedTopPubs = null;

    // Statistics panel
    private VBox statsPanel;
    private Text statsText;

    // Derniers nœuds visualisés par algorithme (pour construire la matrice au clic)
    private final Map<String, List<String>> lastNodesByAlgorithm = new HashMap<>();

    @Override
    public void start(Stage primaryStage) {
        String xmlPath = "C:\\Users\\RPC\\IdeaProjects\\proj\\src\\main\\java\\org\\example\\Fetch\\dblp.xml";
        showLoading(primaryStage, xmlPath);
    }

    private void showLoading(Stage stage, String xmlPath) {
        VBox box = new VBox(20);
        box.setAlignment(Pos.CENTER);
        box.setStyle("-fx-background-color: linear-gradient(to bottom, #2c3e50, #34495e);");

        Text title = new Text("⏳ Chargement du graphe DBLP...");
        title.setFont(Font.font("Segoe UI", 20.0));
        title.setFill(Color.WHITE);

        ProgressBar progressBar = new ProgressBar();
        progressBar.setPrefWidth(400);

        Text progressText = new Text("Initialisation...");
        progressText.setFont(Font.font("Arial", 14));
        progressText.setFill(Color.WHITE);

        box.getChildren().addAll(title, progressBar, progressText);
        Scene scene = new Scene(box, 700, 300);
        stage.setScene(scene);
        stage.setTitle("Chargement - DBLP Visualizer");
        stage.show();

        new Thread(() -> {
            try {
                long startTime = System.currentTimeMillis();

                BipartiteGraphBuilder builder = new BipartiteGraphBuilder();
                builder.parseXML(xmlPath);
                this.graph = builder.getBipartiteGraph();

                cachedTopAuthors = getTopKOptimized(graph.getAuthorVertices(), MAX_DISPLAYED_NODES);
                cachedTopPubs = getTopKOptimized(graph.getPublicationVertices(), MAX_DISPLAYED_NODES);

                long loadTime = System.currentTimeMillis() - startTime;
                System.out.println("Graph loaded in " + loadTime + "ms");

                Platform.runLater(() -> createUI(stage));
            } catch (Exception e) {
                e.printStackTrace();
                Platform.runLater(() -> {
                    new Alert(Alert.AlertType.ERROR, "Erreur: " + e.getMessage()).showAndWait();
                    Platform.exit();
                });
            }
        }).start();
    }

    private void createUI(Stage primaryStage) {
        BorderPane root = new BorderPane();
        root.setStyle("-fx-background-color: #1e2a3a;");

        // Header
        Text title = new Text("🔬 VISUALISATION GRAPHE DBLP + Algorithmes d'Embedding");
        title.setFont(Font.font("Segoe UI", FontWeight.BOLD, 16.0));
        title.setFill(Color.WHITE);

        HBox header = new HBox(title);
        header.setAlignment(Pos.CENTER);
        header.setPadding(new Insets(10));
        header.setStyle("-fx-background-color: linear-gradient(to right, #3498db, #2c3e50);");
        root.setTop(header);

        // Main content (graph + sidebar)
        HBox mainContent = new HBox();
        mainContent.setSpacing(20);
        mainContent.setPadding(new Insets(10));

        // Graph visualization
        VBox graphVisualization = createGraphVisualization();
        HBox.setHgrow(graphVisualization, Priority.ALWAYS);
        graphVisualization.setPrefWidth(1200);

        // Sidebar (statistics) wrapped in ScrollPane
        VBox statsSection = createStatisticsSection();
        ScrollPane statsScroll = new ScrollPane(statsSection);
        statsScroll.setFitToWidth(true);
        statsScroll.setFitToHeight(true);
        statsScroll.setStyle("-fx-background: #2c3e50;");
        statsScroll.setPrefWidth(600);

        mainContent.getChildren().addAll(graphVisualization, statsScroll);
        root.setCenter(mainContent);

        Scene scene = new Scene(root, 1600, 900);
        primaryStage.setScene(scene);
        primaryStage.setTitle("Visualisation Graphe Bipartite DBLP - Algorithmes d'Embedding");
        primaryStage.show();
    }

    private VBox createGraphVisualization() {
        VBox graphContainer = new VBox();
        graphContainer.setSpacing(10);

        TabPane tabPane = new TabPane();
        tabPane.setStyle("-fx-background-color: #1e2a3a;");

        // Tab 1: Graphe
        Tab graphTab = new Tab("🎯 Graphe Bipartite");
        graphTab.setClosable(false);
        VBox graphTabContent = createGraphTab();
        graphTab.setContent(graphTabContent);

        // Tab 2: Embeddings
        Tab embeddingTab = new Tab("🧠 Visualisation des Embeddings");
        embeddingTab.setClosable(false);
        VBox embeddingTabContent = createEmbeddingTab();
        embeddingTab.setContent(embeddingTabContent);

        tabPane.getTabs().addAll(graphTab, embeddingTab);
        graphContainer.getChildren().add(tabPane);

        return graphContainer;
    }

    private VBox createStatisticsSection() {
        VBox statsSection = new VBox();
        statsSection.setSpacing(15);
        statsSection.setPadding(new Insets(10));
        statsSection.setStyle(
                "-fx-background-color: #2c3e50; " +
                        "-fx-border-color: #3498db; " +
                        "-fx-border-width: 2; " +
                        "-fx-border-radius: 8;"
        );

        // Title
        Text statsTitle = new Text("📊 STATISTIQUES ET PRÉDICTIONS");
        statsTitle.setFont(Font.font("Segoe UI", FontWeight.BOLD, 16));
        statsTitle.setFill(Color.WHITE);

        // Main statistics panel
        statsPanel = new VBox(8);
        statsPanel.setStyle("-fx-background-color: #34495e; -fx-padding: 10; -fx-border-radius: 8;");
        statsPanel.setEffect(new DropShadow(10, Color.rgb(0, 0, 0, 0.4)));

        statsText = new Text();
        statsText.setFont(Font.font("Consolas", 11.0));
        statsText.setFill(Color.LIGHTGRAY);
        statsText.setWrappingWidth(560);

        statsPanel.getChildren().addAll(statsText);

        // Algorithm status panel
        VBox algoStatusPanel = createAlgorithmStatusPanel();

        // Link prediction controls
        VBox predictionPanel = createPredictionPanel();

        // Legend panel
        VBox legendPanel = createLegendPanel();

        statsSection.getChildren().addAll(
                statsTitle,
                statsPanel,
                algoStatusPanel,
                predictionPanel,
                legendPanel
        );

        updateStats(0, 0, 0);
        return statsSection;
    }

    private VBox createAlgorithmStatusPanel() {
        VBox statusPanel = new VBox(8);
        statusPanel.setStyle("-fx-background-color: #34495e; -fx-padding: 10; -fx-border-radius: 8;");
        statusPanel.setEffect(new DropShadow(8, Color.rgb(0, 0, 0, 0.3)));

        Text title = new Text("🔧 STATUS DES ALGORITHMES");
        title.setFont(Font.font("Segoe UI", FontWeight.BOLD, 12));
        title.setFill(Color.WHITE);

        HBox node2vecStatus = createAlgorithmStatusBox("Node2Vec", node2vecCalculated, NODE2VEC_COLOR);
        HBox deepwalkStatus = createAlgorithmStatusBox("DeepWalk", deepwalkCalculated, DEEPWALK_COLOR);
        HBox gcnStatus = createAlgorithmStatusBox("GCN", gcnCalculated, GCN_COLOR);
        HBox lineStatus = createAlgorithmStatusBox("LINE", lineCalculated, LINE_COLOR);
        HBox graphSageStatus = createAlgorithmStatusBox("GraphSAGE", graphSageCalculated, GRAPHSAGE_COLOR);

        // Buttons
        HBox calcButtons = new HBox(8);
        calcButtons.setAlignment(Pos.CENTER);
        Button calcNode2Vec = styledButton("🧠 Node2Vec", "#9b59b6");
        Button calcDeepWalk = styledButton("🚶 DeepWalk", "#e74c3c");
        Button calcGCN = styledButton("🕸️ GCN", "#3498db");
        Button calcLINE = styledButton("📈 LINE", "#f39c12");
        Button calcGraphSAGE = styledButton("🌿 GraphSAGE", "#27ae60");

        calcNode2Vec.setOnAction(e -> calculateNode2Vec());
        calcDeepWalk.setOnAction(e -> calculateDeepWalk());
        calcGCN.setOnAction(e -> calculateGCN());
        calcLINE.setOnAction(e -> calculateLINE());
        calcGraphSAGE.setOnAction(e -> calculateGraphSAGE());

        calcButtons.getChildren().addAll(calcNode2Vec, calcDeepWalk, calcGCN, calcLINE, calcGraphSAGE);

        statusPanel.getChildren().addAll(title, node2vecStatus, deepwalkStatus, gcnStatus, lineStatus, graphSageStatus, calcButtons);
        return statusPanel;
    }

    private VBox createPredictionPanel() {
        VBox predictionPanel = new VBox(8);
        predictionPanel.setStyle("-fx-background-color: #34495e; -fx-padding: 10; -fx-border-radius: 8;");
        predictionPanel.setEffect(new DropShadow(8, Color.rgb(0, 0, 0, 0.3)));

        Text title = new Text("🔮 PRÉDICTIONS DE LIENS");
        title.setFont(Font.font("Segoe UI", FontWeight.BOLD, 12));
        title.setFill(Color.WHITE);

        HBox algoSelection = new HBox(5);
        algoSelection.setAlignment(Pos.CENTER);

        ToggleGroup algoGroup = new ToggleGroup();
        RadioButton node2vecRadio = new RadioButton("Node2Vec");
        RadioButton deepwalkRadio = new RadioButton("DeepWalk");
        RadioButton gcnRadio = new RadioButton("GCN");
        RadioButton lineRadio = new RadioButton("LINE");
        RadioButton graphSageRadio = new RadioButton("GraphSAGE");

        node2vecRadio.setToggleGroup(algoGroup);
        deepwalkRadio.setToggleGroup(algoGroup);
        gcnRadio.setToggleGroup(algoGroup);
        lineRadio.setToggleGroup(algoGroup);
        graphSageRadio.setToggleGroup(algoGroup);
        node2vecRadio.setSelected(true);

        String radioStyle = "-fx-text-fill: white; -fx-font-size: 10;";
        node2vecRadio.setStyle(radioStyle);
        deepwalkRadio.setStyle(radioStyle);
        gcnRadio.setStyle(radioStyle);
        lineRadio.setStyle(radioStyle);
        graphSageRadio.setStyle(radioStyle);

        algoSelection.getChildren().addAll(node2vecRadio, deepwalkRadio, gcnRadio, lineRadio, graphSageRadio);

        Button showPredictions = styledButton("👁️ Afficher Prédictions", "#2ecc71");
        Button clearPredictions = styledButton("🗑️ Effacer Prédictions", "#95a5a6");

        showPredictions.setOnAction(e -> {
            String selectedAlgo = "Node2Vec";
            if (deepwalkRadio.isSelected()) selectedAlgo = "DeepWalk";
            else if (gcnRadio.isSelected()) selectedAlgo = "GCN";
            else if (lineRadio.isSelected()) selectedAlgo = "LINE";
            else if (graphSageRadio.isSelected()) selectedAlgo = "GraphSAGE";
            showLinkPredictions(selectedAlgo);
        });

        clearPredictions.setOnAction(e -> clearPredictions());

        predictionPanel.getChildren().addAll(title, algoSelection, showPredictions, clearPredictions);
        return predictionPanel;
    }

    private VBox createLegendPanel() {
        VBox legendPanel = new VBox(8);
        legendPanel.setStyle("-fx-background-color: #34495e; -fx-padding: 10; -fx-border-radius: 8;");
        legendPanel.setEffect(new DropShadow(8, Color.rgb(0, 0, 0, 0.3)));

        Text title = new Text("🎨 LÉGENDE");
        title.setFont(Font.font("Segoe UI", FontWeight.BOLD, 12));
        title.setFill(Color.WHITE);

        legendPanel.getChildren().addAll(title,
                createLegendItem("Auteurs", AUTHOR_COLOR),
                createLegendItem("Publications", PUBLICATION_COLOR),
                createLegendItem("Cachés", DASHED_INDICATOR_COLOR),
                createLegendItem("Prédictions", PREDICTION_COLOR),
                createLegendItem("Node2Vec", NODE2VEC_COLOR),
                createLegendItem("DeepWalk", DEEPWALK_COLOR),
                createLegendItem("GCN", GCN_COLOR),
                createLegendItem("LINE", LINE_COLOR),
                createLegendItem("GraphSAGE", GRAPHSAGE_COLOR));

        return legendPanel;
    }

    private HBox createLegendItem(String text, Color color) {
        HBox legend = new HBox(5);
        legend.setAlignment(Pos.CENTER_LEFT);

        Rectangle colorBox = new Rectangle(12, 12, color);
        colorBox.setStroke(Color.WHITE);
        colorBox.setStrokeWidth(1);

        Text legendText = new Text(text);
        legendText.setFont(Font.font("Segoe UI", 10.0));
        legendText.setFill(Color.WHITE);

        legend.getChildren().addAll(colorBox, legendText);
        return legend;
    }

    private HBox createAlgorithmStatusBox(String algorithm, boolean calculated, Color color) {
        HBox statusBox = new HBox(10);
        statusBox.setAlignment(Pos.CENTER_LEFT);

        Circle statusCircle = new Circle(5);
        statusCircle.setFill(calculated ? Color.LIGHTGREEN : Color.LIGHTGRAY);
        statusCircle.setStroke(Color.WHITE);

        Text statusText = new Text(algorithm + ": " + (calculated ? "✅ Calculé" : "❌ Non calculé"));
        statusText.setFill(Color.WHITE);
        statusText.setFont(Font.font("Segoe UI", FontWeight.BOLD, 10));

        statusBox.getChildren().addAll(statusCircle, statusText);
        return statusBox;
    }

    private VBox createGraphTab() {
        graphPane = new Pane();
        graphPane.setStyle("-fx-background-color: linear-gradient(to bottom, #0f1419, #182233);");
        graphPane.setEffect(new DropShadow(16, Color.rgb(0, 0, 0, 0.35)));
        graphGroup = new Group(graphPane);

        ScrollPane scrollPane = new ScrollPane(graphGroup);
        scrollPane.setPannable(true);
        scrollPane.setFitToWidth(false);
        scrollPane.setFitToHeight(false);
        scrollPane.setStyle("-fx-background: #0f1419; -fx-border-color: #34495e;");
        scrollPane.setCache(true);

        scrollPane.setOnScroll(event -> {
            double delta = event.getDeltaY();
            double scaleFactor = (delta > 0) ? 1.1 : 0.9;
            scale *= scaleFactor;
            scale = Math.max(0.3, Math.min(3.0, scale));
            graphGroup.setScaleX(scale);
            graphGroup.setScaleY(scale);
            event.consume();
        });

        optimizedLayoutGraph();

        HBox ctrlBox = new HBox(8);
        ctrlBox.setAlignment(Pos.CENTER);
        ctrlBox.setPadding(new Insets(8));
        ctrlBox.setStyle("-fx-background-color: #2c3e50; -fx-background-radius: 8;");
        ctrlBox.setEffect(new DropShadow(10, Color.rgb(0, 0, 0, 0.35)));

        Button showEdges = styledButton("🔗 Afficher les Arêtes", "#3498db");
        Button resetBtn = styledButton("🔄 Actualiser", "#3498db");
        Button zoomIn = styledButton("➕ Zoom", "#16a085");
        Button zoomOut = styledButton("➖ Zoom", "#16a085");

        showEdges.setOnAction(e -> drawOptimizedEdges());
        resetBtn.setOnAction(e -> optimizedLayoutGraph());
        zoomIn.setOnAction(e -> { scale *= 1.2; updateScale(); });
        zoomOut.setOnAction(e -> { scale *= 0.8; updateScale(); });

        ctrlBox.getChildren().addAll(showEdges, resetBtn, zoomIn, zoomOut);

        VBox center = new VBox(3, ctrlBox, scrollPane);
        VBox.setVgrow(scrollPane, Priority.ALWAYS);

        return center;
    }

    private VBox createEmbeddingTab() {
        VBox mainContainer = new VBox();
        mainContainer.setSpacing(10);
        mainContainer.setPadding(new Insets(10));
        mainContainer.setStyle("-fx-background-color: #1e2a3a;");

        TabPane algorithmTabPane = new TabPane();
        algorithmTabPane.setStyle("-fx-background-color: #2c3e50;");
        algorithmTabPane.setTabMinWidth(120);
        algorithmTabPane.setTabMinHeight(30);

        algorithmTabPane.getTabs().add(createAlgorithmVisualizationTab("Node2Vec"));
        algorithmTabPane.getTabs().add(createAlgorithmVisualizationTab("DeepWalk"));
        algorithmTabPane.getTabs().add(createAlgorithmVisualizationTab("GCN"));
        algorithmTabPane.getTabs().add(createAlgorithmVisualizationTab("LINE"));
        algorithmTabPane.getTabs().add(createAlgorithmVisualizationTab("GraphSAGE"));

        mainContainer.getChildren().addAll(algorithmTabPane);
        VBox.setVgrow(algorithmTabPane, Priority.ALWAYS);

        return mainContainer;
    }

    private Tab createAlgorithmVisualizationTab(String algorithm) {
        VBox tabContent = new VBox();
        tabContent.setSpacing(10);
        tabContent.setPadding(new Insets(10));
        tabContent.setStyle("-fx-background-color: #1e2a3a;");

        // Augmenter la taille du Pane de visualisation
        Pane visualizationPane = new Pane();
        visualizationPane.setStyle("-fx-background-color: linear-gradient(to bottom, #0f1419, #182233);");
        visualizationPane.setPrefSize(2500, 1500);
        visualizationPane.setMinSize(2500, 1500);
        visualizationPane.setEffect(new DropShadow(16, Color.rgb(0, 0, 0, 0.35)));

        // Ajouter un ScrollPane qui contient le visualizationPane
        ScrollPane embeddingScroll = new ScrollPane(visualizationPane);
        embeddingScroll.setPannable(true);
        embeddingScroll.setFitToWidth(false);
        embeddingScroll.setFitToHeight(false);
        embeddingScroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
        embeddingScroll.setVbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
        embeddingScroll.setStyle("-fx-background: #0f1419; -fx-border-color: #34495e;");
        embeddingScroll.setPrefSize(1500, 700);

        // Gestion du zoom avec la molette de la souris
        embeddingScroll.setOnScroll(event -> {
            double delta = event.getDeltaY();
            double scaleFactor = (delta > 0) ? 1.1 : 0.9;
            double newScaleX = visualizationPane.getScaleX() * scaleFactor;
            double newScaleY = visualizationPane.getScaleY() * scaleFactor;

            // Limiter le zoom
            if (newScaleX >= 0.3 && newScaleX <= 3.0 && newScaleY >= 0.3 && newScaleY <= 3.0) {
                visualizationPane.setScaleX(newScaleX);
                visualizationPane.setScaleY(newScaleY);
            }
            event.consume();
        });

        // Controls avec ScrollPane en paramètre
        HBox controls = createAlgorithmControls(algorithm, visualizationPane, embeddingScroll);

        VBox embeddingLayout = new VBox(10);
        embeddingLayout.getChildren().addAll(controls, embeddingScroll);
        VBox.setVgrow(embeddingScroll, Priority.ALWAYS);

        tabContent.getChildren().add(embeddingLayout);
        VBox.setVgrow(embeddingLayout, Priority.ALWAYS);

        switch (algorithm) {
            case "Node2Vec": this.node2VecPane = visualizationPane; break;
            case "DeepWalk": this.deepWalkPane = visualizationPane; break;
            case "GCN":      this.gcnPane = visualizationPane; break;
            case "LINE":     this.linePane = visualizationPane; break;
            case "GraphSAGE":this.graphSagePane = visualizationPane; break;
        }

        return new Tab(algorithm, tabContent);
    }

    private HBox createAlgorithmControls(String algorithm, Pane visualizationPane, ScrollPane scrollPane) {
        HBox controls = new HBox(10);
        controls.setAlignment(Pos.CENTER);
        controls.setPadding(new Insets(10));
        controls.setStyle("-fx-background-color: #2c3e50; -fx-background-radius: 8;");
        controls.setEffect(new DropShadow(10, Color.rgb(0, 0, 0, 0.35)));

        Button visualizeBtn = styledButton("👁️ Visualiser " + algorithm, getAlgoColorHex(algorithm));
        Button selectRandomBtn = styledButton("🎲 Sélection Aléatoire", "#1abc9c");
        Button clearBtn = styledButton("🗑️ Effacer", "#95a5a6");
        Button similaritiesBtn = styledButton("📊 Détails Similarités (texte)", "#2ecc71");
        Button showMatrixBtn = styledButton("📊 Afficher Matrice Similarité", "#27ae60");
        Button zoomInBtn = styledButton("➕ Zoom", "#16a085");
        Button zoomOutBtn = styledButton("➖ Zoom", "#16a085");
        Button resetZoomBtn = styledButton("🔁 Reset Zoom", "#e74c3c");
        Button centerViewBtn = styledButton("🎯 Centrer Vue", "#9b59b6");

        visualizeBtn.setOnAction(e -> visualizeAlgorithmEmbeddings(algorithm, visualizationPane));
        selectRandomBtn.setOnAction(e -> selectRandomNodesForAlgorithm(algorithm, visualizationPane));
        clearBtn.setOnAction(e -> {
            visualizationPane.getChildren().clear();
            lastNodesByAlgorithm.remove(algorithm);
        });
        similaritiesBtn.setOnAction(e -> showDetailedSimilarities(algorithm));
        showMatrixBtn.setOnAction(e -> showSimilarityMatrixWindow(algorithm));

        // Gestion du zoom avec boutons
        zoomInBtn.setOnAction(e -> {
            double newScaleX = visualizationPane.getScaleX() * 1.2;
            double newScaleY = visualizationPane.getScaleY() * 1.2;
            if (newScaleX <= 3.0) {
                visualizationPane.setScaleX(newScaleX);
                visualizationPane.setScaleY(newScaleY);
            }
        });

        zoomOutBtn.setOnAction(e -> {
            double newScaleX = visualizationPane.getScaleX() * 0.8;
            double newScaleY = visualizationPane.getScaleY() * 0.8;
            if (newScaleX >= 0.3) {
                visualizationPane.setScaleX(newScaleX);
                visualizationPane.setScaleY(newScaleY);
            }
        });

        resetZoomBtn.setOnAction(e -> {
            visualizationPane.setScaleX(1.0);
            visualizationPane.setScaleY(1.0);
        });

        centerViewBtn.setOnAction(e -> {
            // Centrer la vue sur le contenu
            scrollPane.setVvalue(0.5);
            scrollPane.setHvalue(0.5);
        });

        controls.getChildren().addAll(
                visualizeBtn, selectRandomBtn, clearBtn, similaritiesBtn,
                showMatrixBtn, zoomInBtn, zoomOutBtn, resetZoomBtn, centerViewBtn
        );
        return controls;
    }

    // ==================== ALGORITHM CALCULATION METHODS ====================

    private void calculateNode2Vec() {
        if (node2vecCalculated) {
            new Alert(Alert.AlertType.INFORMATION, "Node2Vec déjà calculé!").show();
            return;
        }

        ProgressDialog progressDialog = new ProgressDialog("Calcul Node2Vec");
        progressDialog.show();

        new Thread(() -> {
            try {
                Platform.runLater(() -> progressDialog.updateProgress("Initialisation Node2Vec...", 0.1));

                node2vec = new Node2Vec(graph, 64, 10, 5, 1.0, 1.0, 3, 3);

                Platform.runLater(() -> progressDialog.updateProgress("Calcul des embeddings Node2Vec...", 0.3));
                node2vec.fit();

                Platform.runLater(() -> progressDialog.updateProgress("Récupération des embeddings...", 0.8));
                node2vecEmbeddings = node2vec.getEmbeddings();
                node2vecCalculated = true;

                Platform.runLater(() -> {
                    progressDialog.updateProgress("Terminé!", 1.0);
                    progressDialog.close();
                    updateStats(0, 0, 0);
                    new Alert(Alert.AlertType.INFORMATION,
                            "✅ Node2Vec terminé!\nEmbeddings calculés pour " + node2vecEmbeddings.size() + " nœuds").show();
                });
            } catch (Exception e) {
                Platform.runLater(() -> {
                    progressDialog.close();
                    new Alert(Alert.AlertType.ERROR, "Erreur Node2Vec: " + e.getMessage()).show();
                });
            }
        }).start();
    }

    private void calculateDeepWalk() {
        if (deepwalkCalculated) {
            new Alert(Alert.AlertType.INFORMATION, "DeepWalk déjà calculé!").show();
            return;
        }

        ProgressDialog progressDialog = new ProgressDialog("Calcul DeepWalk");
        progressDialog.show();

        new Thread(() -> {
            try {
                Platform.runLater(() -> progressDialog.updateProgress("Initialisation DeepWalk...", 0.1));

                deepwalk = new DeepWalk(graph, 64, 10, 80, 5, 0.025, 3);

                Platform.runLater(() -> progressDialog.updateProgress("Calcul des embeddings DeepWalk...", 0.3));
                deepwalk.train();

                Platform.runLater(() -> progressDialog.updateProgress("Récupération des embeddings...", 0.8));
                deepwalkEmbeddings = deepwalk.getEmbeddings();
                deepwalkCalculated = true;

                Platform.runLater(() -> {
                    progressDialog.updateProgress("Terminé!", 1.0);
                    progressDialog.close();
                    updateStats(0, 0, 0);
                    new Alert(Alert.AlertType.INFORMATION,
                            "✅ DeepWalk terminé!\nEmbeddings calculés pour " + deepwalkEmbeddings.size() + " nœuds").show();
                });
            } catch (Exception e) {
                Platform.runLater(() -> {
                    progressDialog.close();
                    new Alert(Alert.AlertType.ERROR, "Erreur DeepWalk: " + e.getMessage()).show();
                });
            }
        }).start();
    }

    private void calculateGCN() {
        if (gcnCalculated) {
            new Alert(Alert.AlertType.INFORMATION, "GCN déjà calculé!").show();
            return;
        }

        ProgressDialog progressDialog = new ProgressDialog("Calcul GCN Rapide");
        progressDialog.show();

        new Thread(() -> {
            try {
                Platform.runLater(() -> progressDialog.updateProgress("Initialisation GCN...", 0.1));

                gcn = new GCN(graph, 64, 0.01, 3);

                Platform.runLater(() -> progressDialog.updateProgress("Calcul des embeddings GCN (version rapide)...", 0.3));
                gcn.fitFast();

                Platform.runLater(() -> progressDialog.updateProgress("Récupération des embeddings...", 0.8));
                gcnEmbeddings = gcn.getEmbeddings();
                gcnCalculated = true;

                Platform.runLater(() -> {
                    progressDialog.updateProgress("Terminé!", 1.0);
                    progressDialog.close();
                    updateStats(0, 0, 0);
                    new Alert(Alert.AlertType.INFORMATION,
                            "✅ GCN terminé!\nEmbeddings calculés pour " + gcnEmbeddings.size() + " nœuds").show();
                });
            } catch (Exception e) {
                Platform.runLater(() -> {
                    progressDialog.close();
                    new Alert(Alert.AlertType.ERROR, "Erreur GCN: " + e.getMessage()).show();
                });
            }
        }).start();
    }

    private void calculateLINE() {
        if (lineCalculated) {
            new Alert(Alert.AlertType.INFORMATION, "LINE déjà calculé!").show();
            return;
        }

        ProgressDialog progressDialog = new ProgressDialog("Calcul LINE Rapide");
        progressDialog.show();

        new Thread(() -> {
            try {
                Platform.runLater(() -> progressDialog.updateProgress("Initialisation LINE...", 0.1));

                line = new LINE(graph, 64, 0.025, 5, 5);

                Platform.runLater(() -> progressDialog.updateProgress("Calcul des embeddings LINE (version rapide)...", 0.3));
                line.fitFast();

                Platform.runLater(() -> progressDialog.updateProgress("Récupération des embeddings...", 0.8));
                lineEmbeddings = line.getEmbeddings();
                lineCalculated = true;

                Platform.runLater(() -> {
                    progressDialog.updateProgress("Terminé!", 1.0);
                    progressDialog.close();
                    updateStats(0, 0, 0);
                    new Alert(Alert.AlertType.INFORMATION,
                            "✅ LINE terminé!\nEmbeddings calculés pour " + lineEmbeddings.size() + " nœuds").show();
                });
            } catch (Exception e) {
                Platform.runLater(() -> {
                    progressDialog.close();
                    new Alert(Alert.AlertType.ERROR, "Erreur LINE: " + e.getMessage()).show();
                });
            }
        }).start();
    }

    private void calculateGraphSAGE() {
        if (graphSageCalculated) {
            new Alert(Alert.AlertType.INFORMATION, "GraphSAGE déjà calculé!").show();
            return;
        }

        ProgressDialog progressDialog = new ProgressDialog("Calcul GraphSAGE Rapide");
        progressDialog.show();

        new Thread(() -> {
            try {
                Platform.runLater(() -> progressDialog.updateProgress("Initialisation GraphSAGE...", 0.1));

                graphSage = new GraphSAGE(graph, 64, 10, 0.01, 3);

                Platform.runLater(() -> progressDialog.updateProgress("Calcul des embeddings GraphSAGE (version rapide)...", 0.3));
                graphSage.fitFast();

                Platform.runLater(() -> progressDialog.updateProgress("Récupération des embeddings...", 0.8));
                graphSageEmbeddings = graphSage.getEmbeddings();
                graphSageCalculated = true;

                Platform.runLater(() -> {
                    progressDialog.updateProgress("Terminé!", 1.0);
                    progressDialog.close();
                    updateStats(0, 0, 0);
                    new Alert(Alert.AlertType.INFORMATION,
                            "✅ GraphSAGE terminé!\nEmbeddings calculés pour " + graphSageEmbeddings.size() + " nœuds").show();
                });
            } catch (Exception e) {
                Platform.runLater(() -> {
                    progressDialog.close();
                    new Alert(Alert.AlertType.ERROR, "Erreur GraphSAGE: " + e.getMessage()).show();
                });
            }
        }).start();
    }

    // ==================== EMBEDDING VISUALIZATION METHODS ====================

    private void visualizeAlgorithmEmbeddings(String algorithm, Pane visualizationPane) {
        Map<String, double[]> embeddings = getEmbeddingsForAlgorithm(algorithm);

        if (embeddings == null || embeddings.isEmpty()) {
            new Alert(Alert.AlertType.WARNING,
                    "Veuillez d'abord calculer les embeddings " + algorithm + "!").show();
            return;
        }

        visualizationPane.getChildren().clear();

        List<String> selectedNodes = selectRepresentativeNodes(12);
        lastNodesByAlgorithm.put(algorithm, selectedNodes);

        visualizeSelectedNodesInEmbeddingSpace(selectedNodes, embeddings, algorithm, visualizationPane);
    }

    private void selectRandomNodesForAlgorithm(String algorithm, Pane visualizationPane) {
        Map<String, double[]> embeddings = getEmbeddingsForAlgorithm(algorithm);

        if (embeddings == null || embeddings.isEmpty()) {
            new Alert(Alert.AlertType.WARNING,
                    "Veuillez d'abord calculer les embeddings " + algorithm + "!").show();
            return;
        }

        visualizationPane.getChildren().clear();

        List<String> randomNodes = selectRandomNodes(12);
        lastNodesByAlgorithm.put(algorithm, randomNodes);

        visualizeSelectedNodesInEmbeddingSpace(randomNodes, embeddings, algorithm, visualizationPane);
    }

    private Map<String, double[]> getEmbeddingsForAlgorithm(String algorithm) {
        switch (algorithm) {
            case "Node2Vec": return node2vecEmbeddings;
            case "DeepWalk": return deepwalkEmbeddings;
            case "GCN":      return gcnEmbeddings;
            case "LINE":     return lineEmbeddings;
            case "GraphSAGE":return graphSageEmbeddings;
            default: return null;
        }
    }

    // ==================== LINK PREDICTION METHODS ====================

    private void showLinkPredictions(String algorithm) {
        Map<String, double[]> embeddings = getEmbeddingsForAlgorithm(algorithm);

        if (embeddings == null || embeddings.isEmpty()) {
            new Alert(Alert.AlertType.WARNING,
                    "Veuillez d'abord calculer les embeddings " + algorithm + "!").show();
            return;
        }

        List<String> displayedNodes = new ArrayList<>(nodeCircles.keySet());
        if (displayedNodes.size() < 2) {
            new Alert(Alert.AlertType.WARNING, "Pas assez de nœuds affichés pour les prédictions!").show();
            return;
        }

        clearPredictions();

        int maxPredictions = 10;
        List<Prediction> predictions = new ArrayList<>();

        List<String> displayedAuthors = displayedNodes.stream()
                .filter(node -> graph.getAuthorVertices().contains(node))
                .collect(Collectors.toList());

        List<String> displayedPublications = displayedNodes.stream()
                .filter(node -> graph.getPublicationVertices().contains(node))
                .collect(Collectors.toList());

        for (String author : displayedAuthors) {
            for (String publication : displayedPublications) {
                if (!graph.getNeighbors(author).contains(publication)) {
                    double similarity = cosineSimilarity(embeddings.get(author), embeddings.get(publication));
                    double probability = (1 + similarity) / 2;

                    if (probability > 0.3) {
                        predictions.add(new Prediction(author, publication, probability, similarity));
                    }
                }
            }
        }

        predictions.sort((p1, p2) -> Double.compare(p2.probability, p1.probability));

        int drawn = 0;
        for (Prediction pred : predictions) {
            if (drawn >= maxPredictions) break;
            Circle circle1 = nodeCircles.get(pred.node1);
            Circle circle2 = nodeCircles.get(pred.node2);
            if (circle1 == null || circle2 == null) continue;

            Line predictionLine = new Line(
                    circle1.getCenterX(), circle1.getCenterY(),
                    circle2.getCenterX(), circle2.getCenterY()
            );

            predictionLine.setStroke(PREDICTION_COLOR);
            predictionLine.setStrokeWidth(2);
            predictionLine.getStrokeDashArray().addAll(5.0, 5.0);

            String tooltipText = String.format(
                    "🔮 Prédiction de lien %s\n%s ↔ %s\nProbabilité: %.3f\nSimilarité: %.4f",
                    algorithm, pred.node1, pred.node2, pred.probability, pred.similarity
            );
            Tooltip.install(predictionLine, new Tooltip(tooltipText));

            double midX = (circle1.getCenterX() + circle2.getCenterX()) / 2;
            double midY = (circle1.getCenterY() + circle2.getCenterY()) / 2;

            Text probabilityLabel = new Text(String.format("%.3f", pred.probability));
            probabilityLabel.setFont(Font.font("Consolas", FontWeight.BOLD, 10));
            probabilityLabel.setFill(Color.LIGHTGREEN);
            probabilityLabel.setX(midX - 15);
            probabilityLabel.setY(midY - 5);

            graphPane.getChildren().addAll(predictionLine, probabilityLabel);
            drawn++;
        }
    }

    private void clearPredictions() {
        graphPane.getChildren().removeIf(n -> n instanceof Line && ((Line)n).getStroke() == PREDICTION_COLOR);
        graphPane.getChildren().removeIf(n -> n instanceof Text && ((Text)n).getFill() == Color.LIGHTGREEN);
    }

    // ==================== GRAPH LAYOUT METHODS ====================

    private List<String> getTopKOptimized(Set<String> nodes, int k) {
        if (nodes.isEmpty() || k <= 0) return new ArrayList<>();

        PriorityQueue<Map.Entry<String, Integer>> pq =
                new PriorityQueue<>(k, Map.Entry.comparingByValue());

        Iterator<String> iterator = nodes.iterator();
        int count = 0;

        while (iterator.hasNext() && count < k) {
            String node = iterator.next();
            pq.offer(new AbstractMap.SimpleEntry<>(node, graph.getDegree(node)));
            count++;
        }

        while (iterator.hasNext()) {
            String node = iterator.next();
            int degree = graph.getDegree(node);
            if (degree > pq.peek().getValue()) {
                pq.poll();
                pq.offer(new AbstractMap.SimpleEntry<>(node, degree));
            }
        }

        List<String> result = new ArrayList<>(k);
        while (!pq.isEmpty()) result.add(pq.poll().getKey());
        Collections.reverse(result);
        return result;
    }

    private void optimizedLayoutGraph() {
        if (graph == null) return;

        long startTime = System.currentTimeMillis();

        graphPane.getChildren().clear();
        nodeCircles.clear();
        nodeLabels.clear();
        edgesDrawn = false;

        List<String> topAuthors = cachedTopAuthors;
        List<String> topPubs = cachedTopPubs;

        double paneWidth = 1200;
        double paneHeight = 600;
        graphPane.setPrefSize(paneWidth, paneHeight);

        placeNodesOnCircle(topAuthors, paneWidth * 0.25, paneHeight / 2, 160, true);
        placeNodesOnCircle(topPubs, paneWidth * 0.75, paneHeight / 2, 160, false);

        addHiddenNodesIndicators(topAuthors, graph.getAuthorVertices(), paneWidth * 0.25, paneHeight / 2, 200, true);
        addHiddenNodesIndicators(topPubs, graph.getPublicationVertices(), paneWidth * 0.75, paneHeight / 2, 200, false);

        long layoutTime = System.currentTimeMillis() - startTime;
        updateStats(layoutTime, 0, 0);
    }

    private void placeNodesOnCircle(List<String> nodes, double centerX, double centerY, double radius, boolean isAuthor) {
        if (nodes.isEmpty()) return;

        int n = nodes.size();
        double angleStep = 2 * Math.PI / n;

        for (int i = 0; i < n; i++) {
            double angle = i * angleStep;
            double x = centerX + radius * Math.cos(angle);
            double y = centerY + radius * Math.sin(angle);
            createEnhancedNode(nodes.get(i), x, y, isAuthor, i);
        }
    }

    private void addHiddenNodesIndicators(List<String> displayedNodes, Set<String> allNodes,
                                          double centerX, double centerY, double radius, boolean isAuthor) {
        int hiddenCount = allNodes.size() - displayedNodes.size();
        if (hiddenCount <= 0) return;

        Circle hiddenCircle = new Circle(centerX, centerY, radius);
        hiddenCircle.setFill(Color.TRANSPARENT);
        hiddenCircle.setStroke(DASHED_INDICATOR_COLOR);
        hiddenCircle.setStrokeWidth(2);
        hiddenCircle.getStrokeDashArray().addAll(10.0, 5.0);

        Text hiddenText = new Text("+" + hiddenCount + " " + (isAuthor ? "auteurs" : "publications"));
        hiddenText.setFont(Font.font("Segoe UI", 10.0));
        hiddenText.setFill(DASHED_INDICATOR_COLOR);
        hiddenText.setX(centerX - hiddenText.getLayoutBounds().getWidth() / 2);
        hiddenText.setY(centerY + radius + 25);

        Tooltip tip = new Tooltip(hiddenCount + " " + (isAuthor ? "auteurs" : "publications") + " cachés");
        Tooltip.install(hiddenCircle, tip);
        Tooltip.install(hiddenText, tip);

        graphPane.getChildren().addAll(hiddenCircle, hiddenText);
    }

    private void createEnhancedNode(String id, double x, double y, boolean isAuthor, int index) {
        int deg = graph.getDegree(id);
        double radius = 10 + Math.min(8, Math.log(deg + 1));

        Circle circle = new Circle(x, y, radius);
        circle.setFill(isAuthor ? AUTHOR_COLOR : PUBLICATION_COLOR);
        circle.setStroke(isAuthor ? AUTHOR_STROKE : PUBLICATION_STROKE);
        circle.setStrokeWidth(1.5);
        circle.setEffect(new DropShadow(10, Color.rgb(0, 0, 0, 0.5)));

        Glow glow = new Glow(0.0);
        circle.setEffect(glow);
        circle.setOnMouseEntered(e -> glow.setLevel(0.5));
        circle.setOnMouseExited(e -> glow.setLevel(0.0));

        String tooltipText = String.format("%s\nDegré: %d\nConnections: %d",
                id, deg, graph.getNeighbors(id).size());
        Tooltip tip = new Tooltip(tooltipText);
        Tooltip.install(circle, tip);

        graphPane.getChildren().add(circle);
        nodeCircles.put(id, circle);

        String displayName = getCompleteDisplayName(id, isAuthor);
        Text label = new Text(displayName);
        label.setFont(Font.font("Segoe UI", isAuthor ? 9.0 : 8.5));
        label.setFill(isAuthor ? AUTHOR_LABEL : PUBLICATION_LABEL);

        double labelX = x - label.getLayoutBounds().getWidth() / 2;
        double labelY = y + radius + 18;

        label.setX(labelX);
        label.setY(labelY);

        graphPane.getChildren().add(label);
        nodeLabels.put(id, label);
    }

    private String getCompleteDisplayName(String id, boolean isAuthor) {
        if (isAuthor) {
            if (id.length() > 20) {
                String[] parts = id.split("\\s+");
                if (parts.length >= 2) {
                    return parts[0] + "\n" + parts[parts.length - 1];
                }
                return id.substring(0, 10) + "...\n" + id.substring(id.length() - 8);
            }
            return id;
        } else {
            if (id.length() > 25) {
                String[] words = id.split("\\s+");
                if (words.length >= 4) {
                    return words[0] + " " + words[1] + "...\n" + words[words.length - 2] + " " + words[words.length - 1];
                } else if (words.length >= 2) {
                    int mid = words.length / 2;
                    StringBuilder firstLine = new StringBuilder();
                    StringBuilder secondLine = new StringBuilder();

                    for (int i = 0; i < mid; i++) firstLine.append(words[i]).append(" ");
                    for (int i = mid; i < words.length; i++) secondLine.append(words[i]).append(" ");

                    return firstLine.toString().trim() + "\n" + secondLine.toString().trim();
                }
                return id.substring(0, 12) + "...\n" + id.substring(id.length() - 12);
            }
            return id;
        }
    }

    private void drawOptimizedEdges() {
        if (edgesDrawn) {
            animateEdges();
            return;
        }

        long startTime = System.currentTimeMillis();
        graphPane.getChildren().removeIf(n -> n instanceof Line);

        Set<String> displayed = nodeCircles.keySet();
        List<Line> edgesToDraw = new ArrayList<>();
        int drawn = 0;

        for (String author : displayed) {
            if (!graph.getAuthorVertices().contains(author)) continue;
            if (drawn >= MAX_DISPLAYED_EDGES) break;

            for (String pub : graph.getNeighbors(author)) {
                if (displayed.contains(pub)) {
                    Circle a = nodeCircles.get(author);
                    Circle p = nodeCircles.get(pub);
                    if (a != null && p != null) {
                        Line line = createOptimizedEdge(a, p);
                        edgesToDraw.add(line);
                        drawn++;
                        if (drawn >= MAX_DISPLAYED_EDGES) break;
                    }
                }
            }
        }

        graphPane.getChildren().addAll(0, edgesToDraw);
        edgesDrawn = true;

        long edgeTime = System.currentTimeMillis() - startTime;
        updateStats(0, drawn, edgeTime);
    }

    private Line createOptimizedEdge(Circle start, Circle end) {
        Line line = new Line(start.getCenterX(), start.getCenterY(),
                end.getCenterX(), end.getCenterY());
        line.setStroke(EDGE_COLOR);
        line.setStrokeWidth(0.8);
        line.setStrokeLineCap(StrokeLineCap.ROUND);
        return line;
    }

    private void animateEdges() {
        graphPane.getChildren().stream()
                .filter(n -> n instanceof Line)
                .limit(10)
                .forEach(n -> {
                    FadeTransition ft = new FadeTransition(Duration.millis(400), n);
                    ft.setFromValue(0.3);
                    ft.setToValue(1.0);
                    ft.setCycleCount(2);
                    ft.setAutoReverse(true);
                    ft.play();
                });
    }

    private void updateScale() {
        scale = Math.max(0.3, Math.min(3.0, scale));
        graphGroup.setScaleX(scale);
        graphGroup.setScaleY(scale);
    }

    // ==================== EMBEDDING VISUALIZATION METHODS ====================

    private List<String> selectRepresentativeNodes(int count) {
        List<String> selected = new ArrayList<>();

        List<String> topAuthors = getTopKOptimized(graph.getAuthorVertices(), count/2);
        List<String> topPubs = getTopKOptimized(graph.getPublicationVertices(), count/2);

        selected.addAll(topAuthors);
        selected.addAll(topPubs);

        if (selected.size() < count) {
            Set<String> allNodes = new HashSet<>(graph.getAuthorVertices());
            allNodes.addAll(graph.getPublicationVertices());
            allNodes.removeAll(selected);

            List<String> remaining = new ArrayList<>(allNodes);
            Collections.shuffle(remaining);
            int needed = count - selected.size();
            selected.addAll(remaining.subList(0, Math.min(needed, remaining.size())));
        }

        return selected.subList(0, Math.min(count, selected.size()));
    }

    private List<String> selectRandomNodes(int count) {
        Set<String> allNodes = new HashSet<>(graph.getAuthorVertices());
        allNodes.addAll(graph.getPublicationVertices());

        List<String> nodeList = new ArrayList<>(allNodes);
        Collections.shuffle(nodeList);

        return nodeList.subList(0, Math.min(count, nodeList.size()));
    }

    private void visualizeSelectedNodesInEmbeddingSpace(List<String> nodes, Map<String, double[]> embeddings,
                                                        String algorithm, Pane visualizationPane) {
        // Nettoyer le pane
        visualizationPane.getChildren().clear();

        // Utiliser l'espace disponible
        double paneWidth = 2500;
        double paneHeight = 1500;
        visualizationPane.setPrefSize(paneWidth, paneHeight);
        visualizationPane.setMinSize(paneWidth, paneHeight);

        displayVectorInfo(nodes, embeddings, algorithm, visualizationPane, paneWidth, paneHeight);

        Map<String, double[]> projectedEmbeddings = projectTo2DWithPCA(nodes, embeddings);

        // Positionner le système de coordonnées au centre
        drawEnhancedEmbeddingSpace(projectedEmbeddings, nodes, embeddings, algorithm, visualizationPane, paneWidth, paneHeight);

        addCoordinatePanel(nodes, embeddings, algorithm, visualizationPane, paneWidth, paneHeight);

        addLinkPredictions(nodes, projectedEmbeddings, embeddings, algorithm, visualizationPane, paneWidth, paneHeight);

        addInstructionsPanel(visualizationPane, algorithm, paneWidth, paneHeight);
    }

    private Map<String, double[]> projectTo2DWithPCA(List<String> nodes, Map<String, double[]> embeddings) {
        Map<String, double[]> projected = new HashMap<>();

        for (String node : nodes) {
            double[] embedding = embeddings.get(node);
            if (embedding != null && embedding.length >= 2) {
                double[] projection = new double[2];

                for (int i = 0; i < embedding.length; i++) {
                    double weightX = Math.sin(i * 0.1) * 0.5 + 0.5;
                    double weightY = Math.cos(i * 0.2) * 0.5 + 0.5;
                    projection[0] += embedding[i] * weightX;
                    projection[1] += embedding[i] * weightY;
                }

                double norm = Math.sqrt(projection[0]*projection[0] + projection[1]*projection[1]);
                if (norm > 0) {
                    projection[0] /= norm;
                    projection[1] /= norm;
                }

                projected.put(node, projection);
            }
        }

        return projected;
    }

    private void displayVectorInfo(List<String> nodes, Map<String, double[]> embeddings,
                                   String algorithm, Pane visualizationPane, double paneWidth, double paneHeight) {
        VBox infoPanel = new VBox(5);
        infoPanel.setStyle("-fx-background-color: rgba(0,0,0,0.75); -fx-padding: 10; -fx-border-color: #3498db; -fx-border-width: 1; -fx-background-radius: 8;");
        infoPanel.setEffect(new DropShadow(8, Color.rgb(0,0,0,0.5)));
        infoPanel.setLayoutX(20);
        infoPanel.setLayoutY(20);
        infoPanel.setPrefWidth(350);
        infoPanel.setMaxHeight(paneHeight - 40);

        Text title = new Text("📐 VECTEURS 64D - " + algorithm);
        title.setFill(Color.YELLOW);
        title.setFont(Font.font("Segoe UI", FontWeight.BOLD, 14));

        Color algoColor = getAlgorithmColor(algorithm);
        Text algoText = new Text("Algorithme: " + algorithm);
        algoText.setFill(algoColor);
        algoText.setFont(Font.font("Segoe UI", FontWeight.BOLD, 12));

        infoPanel.getChildren().addAll(title, algoText);

        if (!nodes.isEmpty()) {
            String sampleNode = nodes.get(0);
            double[] sampleVector = embeddings.get(sampleNode);
            if (sampleVector != null) {
                Text vectorInfo = new Text(String.format("Chaque vecteur: %d dimensions\nExemple: %s\n3 premières coordonnées: %.3f, %.3f, %.3f",
                        sampleVector.length, sampleNode, sampleVector[0], sampleVector[1], sampleVector[2]));
                vectorInfo.setFill(Color.LIGHTGREEN);
                vectorInfo.setFont(Font.font("Consolas", 10));
                vectorInfo.setWrappingWidth(330);
                infoPanel.getChildren().add(vectorInfo);
            }
        }

        if (nodes.size() >= 2) {
            infoPanel.getChildren().add(new Text(" "));
            Text simTitle = new Text("🔍 SIMILARITÉS COSINUS:");
            simTitle.setFill(Color.CYAN);
            simTitle.setFont(Font.font("Segoe UI", FontWeight.BOLD, 12));
            infoPanel.getChildren().add(simTitle);

            for (int i = 0; i < Math.min(3, nodes.size()); i++) {
                for (int j = i + 1; j < Math.min(4, nodes.size()); j++) {
                    String node1 = nodes.get(i);
                    String node2 = nodes.get(j);
                    double similarity = cosineSimilarity(embeddings.get(node1), embeddings.get(node2));
                    double angle = Math.acos(Math.max(-1, Math.min(1, similarity))) * 180 / Math.PI;
                    double probability = (1 + similarity) / 2;

                    Text simText = new Text(String.format("  %s ↔ %s\n    Similarité: %.4f\n    Angle: %.1f°\n    Probabilité: %.2f\n",
                            getShortName(node1, graph.getAuthorVertices().contains(node1)),
                            getShortName(node2, graph.getAuthorVertices().contains(node2)),
                            similarity, angle, probability));
                    simText.setFill(Color.WHITE);
                    simText.setFont(Font.font("Consolas", 9));
                    simText.setWrappingWidth(330);
                    infoPanel.getChildren().add(simText);
                }
            }
        }

        visualizationPane.getChildren().add(infoPanel);
    }

    private void drawEnhancedEmbeddingSpace(Map<String, double[]> projectedEmbeddings, List<String> nodes,
                                            Map<String, double[]> embeddings, String algorithm,
                                            Pane visualizationPane, double paneWidth, double paneHeight) {
        double centerX = paneWidth / 2;
        double centerY = paneHeight / 2;
        double scale = Math.min(paneWidth, paneHeight) * 0.15;

        drawEnhancedCoordinateSystem(centerX, centerY, scale, visualizationPane, paneWidth, paneHeight);

        Text algoTitle = new Text(algorithm + " - Espace 2D (Projection PCA)");
        algoTitle.setFont(Font.font("Segoe UI", FontWeight.BOLD, 16));
        algoTitle.setFill(getAlgorithmColor(algorithm));
        algoTitle.setX(centerX - algoTitle.getLayoutBounds().getWidth() / 2);
        algoTitle.setY(centerY - scale - 80);
        visualizationPane.getChildren().add(algoTitle);

        for (String node : nodes) {
            double[] projection = projectedEmbeddings.get(node);
            if (projection != null) {
                double x = centerX + projection[0] * scale;
                double y = centerY + projection[1] * scale;

                boolean isAuthor = graph.getAuthorVertices().contains(node);
                drawVectorPoint(node, x, y, isAuthor, projection, embeddings.get(node), algorithm, visualizationPane);
            }
        }

        drawVectorLines(projectedEmbeddings, nodes, centerX, centerY, scale, visualizationPane);
    }

    private void drawEnhancedCoordinateSystem(double centerX, double centerY, double scale,
                                              Pane visualizationPane, double paneWidth, double paneHeight) {
        Line xAxis = new Line(centerX - scale - 50, centerY, centerX + scale + 50, centerY);
        Line yAxis = new Line(centerX, centerY - scale - 50, centerX, centerY + scale + 50);

        xAxis.setStroke(Color.GRAY);
        yAxis.setStroke(Color.GRAY);
        xAxis.setStrokeWidth(1.5);
        yAxis.setStrokeWidth(1.5);

        for (int i = -1; i <= 1; i++) {
            if (i == 0) continue;

            Line vLine = new Line(centerX + i * scale, centerY - scale - 30,
                    centerX + i * scale, centerY + scale + 30);
            vLine.setStroke(Color.rgb(50, 50, 50, 0.5));
            vLine.setStrokeWidth(0.5);
            vLine.getStrokeDashArray().addAll(2.0, 4.0);

            Line hLine = new Line(centerX - scale - 30, centerY + i * scale,
                    centerX + scale + 30, centerY + i * scale);
            hLine.setStroke(Color.rgb(50, 50, 50, 0.5));
            hLine.setStrokeWidth(0.5);
            hLine.getStrokeDashArray().addAll(2.0, 4.0);

            visualizationPane.getChildren().addAll(vLine, hLine);
        }

        for (int i = -1; i <= 1; i++) {
            if (i != 0) {
                Text xLabel = new Text(String.format("%.1f", (double)i));
                xLabel.setFill(Color.LIGHTGRAY);
                xLabel.setFont(Font.font("Consolas", 10));
                xLabel.setX(centerX + i * scale - 8);
                xLabel.setY(centerY + 20);

                Text yLabel = new Text(String.format("%.1f", (double)i));
                yLabel.setFill(Color.LIGHTGRAY);
                yLabel.setFont(Font.font("Consolas", 10));
                yLabel.setX(centerX - 25);
                yLabel.setY(centerY + i * scale + 4);

                visualizationPane.getChildren().addAll(xLabel, yLabel);
            }
        }

        Text xLabel = new Text("Dimension X (Projected)");
        Text yLabel = new Text("Dimension Y (Projected)");
        xLabel.setFill(Color.LIGHTGRAY);
        yLabel.setFill(Color.LIGHTGRAY);
        xLabel.setX(centerX + scale + 10);
        xLabel.setY(centerY + 15);
        yLabel.setX(centerX - 100);
        yLabel.setY(centerY - scale - 10);

        visualizationPane.getChildren().addAll(xAxis, yAxis, xLabel, yLabel);
    }

    private void drawVectorPoint(String nodeId, double x, double y, boolean isAuthor,
                                 double[] projection, double[] fullVector, String algorithm, Pane visualizationPane) {
        double radius = 6;

        Circle point = new Circle(x, y, radius);
        point.setFill(getAlgorithmColor(algorithm));
        point.setStroke(Color.WHITE);
        point.setStrokeWidth(1.5);

        Glow glow = new Glow(0.0);
        point.setEffect(glow);
        point.setOnMouseEntered(e -> glow.setLevel(0.6));
        point.setOnMouseExited(e -> glow.setLevel(0.0));

        StringBuilder vectorInfo = new StringBuilder();
        vectorInfo.append(nodeId).append("\n");
        vectorInfo.append("Type: ").append(isAuthor ? "Auteur" : "Publication").append("\n");
        vectorInfo.append("Algorithme: ").append(algorithm).append("\n");
        vectorInfo.append("Projection 2D: (").append(String.format("%.3f, %.3f", projection[0], projection[1])).append(")\n");
        vectorInfo.append("Vecteur complet (64D):\n");

        for (int i = 0; i < Math.min(6, fullVector.length); i++) {
            vectorInfo.append(String.format("  [%2d]: %7.4f\n", i, fullVector[i]));
        }
        vectorInfo.append("  ... et ").append(fullVector.length - 6).append(" dimensions supplémentaires");

        Tooltip tip = new Tooltip(vectorInfo.toString());
        Tooltip.install(point, tip);

        String shortName = getShortName(nodeId, isAuthor);
        Text label = new Text(shortName + String.format("\n(%.2f,%.2f)", projection[0], projection[1]));
        label.setFont(Font.font("Consolas", 8.0));
        label.setFill(Color.LIGHTGRAY);
        label.setTextAlignment(TextAlignment.CENTER);
        label.setX(x - label.getLayoutBounds().getWidth() / 2);
        label.setY(y + radius + 25);

        visualizationPane.getChildren().addAll(point, label);
    }

    private void drawVectorLines(Map<String, double[]> projectedEmbeddings, List<String> nodes,
                                 double centerX, double centerY, double scale, Pane visualizationPane) {
        for (String node : nodes) {
            double[] projection = projectedEmbeddings.get(node);
            if (projection != null) {
                double x = centerX + projection[0] * scale;
                double y = centerY + projection[1] * scale;

                Line vectorLine = new Line(centerX, centerY, x, y);
                vectorLine.setStroke(Color.rgb(100, 100, 100, 0.4));
                vectorLine.setStrokeWidth(1);
                vectorLine.getStrokeDashArray().addAll(3.0, 2.0);

                visualizationPane.getChildren().add(0, vectorLine);
            }
        }
    }

    private void addCoordinatePanel(List<String> nodes, Map<String, double[]> embeddings,
                                    String algorithm, Pane visualizationPane, double paneWidth, double paneHeight) {
        VBox coordPanel = new VBox(5);
        coordPanel.setStyle("-fx-background-color: rgba(0,0,0,0.7); -fx-padding: 10; -fx-border-color: #e74c3c; -fx-border-width: 1; -fx-background-radius: 8;");
        coordPanel.setEffect(new DropShadow(8, Color.rgb(0,0,0,0.5)));
        coordPanel.setLayoutX(paneWidth - 380);
        coordPanel.setLayoutY(50);
        coordPanel.setPrefWidth(350);
        coordPanel.setMaxHeight(400);

        Text title = new Text("🎯 COORDONNÉES VECTEURS (64D)");
        title.setFill(Color.ORANGE);
        title.setFont(Font.font("Segoe UI", FontWeight.BOLD, 12));

        coordPanel.getChildren().add(title);

        Text algoInfo = new Text("Algorithme: " + algorithm);
        algoInfo.setFill(getAlgorithmColor(algorithm));
        algoInfo.setFont(Font.font("Segoe UI", FontWeight.BOLD, 10));
        coordPanel.getChildren().add(algoInfo);

        for (int i = 0; i < Math.min(3, nodes.size()); i++) {
            String node = nodes.get(i);
            double[] vector = embeddings.get(node);

            if (vector != null) {
                Text nodeTitle = new Text(getShortName(node, graph.getAuthorVertices().contains(node)) + ":");
                nodeTitle.setFill(Color.LIGHTCYAN);
                nodeTitle.setFont(Font.font("Segoe UI", FontWeight.BOLD, 10));
                coordPanel.getChildren().add(nodeTitle);

                StringBuilder coords = new StringBuilder("  ");
                for (int j = 0; j < Math.min(8, vector.length); j++) {
                    coords.append(String.format("[%d]:%7.4f ", j, vector[j]));
                    if ((j + 1) % 4 == 0) coords.append("\n  ");
                }

                Text coordText = new Text(coords.toString());
                coordText.setFill(Color.WHITE);
                coordText.setFont(Font.font("Consolas", 8));
                coordText.setWrappingWidth(330);
                coordPanel.getChildren().add(coordText);

                if (i < Math.min(3, nodes.size()) - 1) {
                    coordPanel.getChildren().add(new Text(" "));
                }
            }
        }

        visualizationPane.getChildren().add(coordPanel);
    }

    private void addInstructionsPanel(Pane visualizationPane, String algorithm, double paneWidth, double paneHeight) {
        VBox instructions = new VBox(10);
        instructions.setStyle("-fx-background-color: rgba(30, 30, 30, 0.85); -fx-padding: 15; -fx-border-color: #3498db; -fx-border-width: 2; -fx-background-radius: 10;");
        instructions.setEffect(new DropShadow(10, Color.rgb(0, 0, 0, 0.5)));
        instructions.setLayoutX(paneWidth - 350);
        instructions.setLayoutY(paneHeight - 250);
        instructions.setPrefWidth(300);

        Text title = new Text("📋 Instructions");
        title.setFont(Font.font("Segoe UI", FontWeight.BOLD, 14));
        title.setFill(Color.YELLOW);

        TextArea instructionsText = new TextArea();
        instructionsText.setEditable(false);
        instructionsText.setWrapText(true);
        instructionsText.setText(
                "Navigation :\n" +
                        "• Défilement : Barres de défilement\n" +
                        "• Zoom : Molette souris\n" +
                        "• Zoom In/Out : Boutons +/- ou Ctrl+Molette\n" +
                        "• Recentrer : Bouton 🎯\n\n" +
                        "Visualisation :\n" +
                        "• Survolez les points pour détails\n" +
                        "• Cliquez sur 📊 pour matrice\n" +
                        "• 🎲 : Sélection aléatoire\n\n" +
                        "Algorithmes :\n" +
                        "• Node2Vec : Bleu\n" +
                        "• DeepWalk : Violet\n" +
                        "• GCN : Rouge\n" +
                        "• LINE : Orange\n" +
                        "• GraphSAGE : Vert"
        );
        instructionsText.setStyle("-fx-control-inner-background: #2c3e50; -fx-text-fill: white; -fx-font-size: 11;");
        instructionsText.setPrefHeight(200);

        instructions.getChildren().addAll(title, instructionsText);
        visualizationPane.getChildren().add(instructions);
    }

    // ==================== SIMILARITY MATRIX WINDOW ====================

    private void showSimilarityMatrixWindow(String algorithm) {
        Map<String, double[]> embeddings = getEmbeddingsForAlgorithm(algorithm);

        if (embeddings == null || embeddings.isEmpty()) {
            new Alert(Alert.AlertType.WARNING,
                    "Veuillez d'abord calculer les embeddings " + algorithm + "!").show();
            return;
        }

        List<String> nodes = lastNodesByAlgorithm.getOrDefault(algorithm, selectRepresentativeNodes(12));

        VBox matrixPanel = buildSimilarityMatrixPanel(nodes, embeddings, algorithm);

        ScrollPane scroll = new ScrollPane(matrixPanel);
        scroll.setFitToWidth(true);
        scroll.setPrefSize(900, 600);
        scroll.setStyle("-fx-background: #1c1c1c; -fx-border-color: #34495e;");

        Stage stage = new Stage();
        stage.setTitle("📊 Matrice de Similarité - " + algorithm);
        BorderPane root = new BorderPane(scroll);
        root.setStyle("-fx-background-color: #1e2a3a;");
        root.setPadding(new Insets(12));

        Scene scene = new Scene(root, 950, 650);
        stage.setScene(scene);
        stage.show();
    }

    private VBox buildSimilarityMatrixPanel(List<String> nodes, Map<String, double[]> embeddings, String algorithm) {
        VBox matrixPanel = new VBox(10);
        matrixPanel.setStyle("-fx-background-color: rgba(0,0,0,0.7); -fx-padding: 12; -fx-border-color: #2ecc71; -fx-border-width: 1; -fx-background-radius: 8;");
        matrixPanel.setEffect(new DropShadow(8, Color.rgb(0, 0, 0, 0.4)));

        Label title = new Label("📊 MATRICE DE SIMILARITÉ - " + algorithm);
        title.setTextFill(Color.LIGHTGREEN);
        title.setFont(Font.font("Segoe UI", FontWeight.BOLD, 14));
        matrixPanel.getChildren().add(title);

        int matrixSize = Math.min(8, nodes.size());

        HBox headerRow = new HBox(12);
        Region spacer = new Region();
        spacer.setPrefWidth(70);
        headerRow.getChildren().add(spacer);
        for (int i = 0; i < matrixSize; i++) {
            String shortName = getShortNameForMatrix(nodes.get(i), graph.getAuthorVertices().contains(nodes.get(i)));
            Label header = new Label(shortName);
            header.setTextFill(Color.YELLOW);
            header.setFont(Font.font("Consolas", 10));
            header.setPrefWidth(50);
            header.setRotate(-45);
            headerRow.getChildren().add(header);
        }
        matrixPanel.getChildren().add(headerRow);

        for (int i = 0; i < matrixSize; i++) {
            HBox row = new HBox(8);

            String rowName = getShortNameForMatrix(nodes.get(i), graph.getAuthorVertices().contains(nodes.get(i)));
            Label rowHeader = new Label(rowName);
            rowHeader.setTextFill(Color.YELLOW);
            rowHeader.setFont(Font.font("Consolas", 10));
            rowHeader.setPrefWidth(60);
            row.getChildren().add(rowHeader);

            for (int j = 0; j < matrixSize; j++) {
                double similarity = cosineSimilarity(embeddings.get(nodes.get(i)), embeddings.get(nodes.get(j)));
                double probability = (1 + similarity) / 2;

                Label simLabel = new Label(String.format("%.2f", similarity));
                simLabel.setTooltip(new Tooltip(String.format(
                        "%s ↔ %s\nSimilarité: %.4f\nProbabilité: %.3f\nAlgo: %s",
                        nodes.get(i), nodes.get(j), similarity, probability, algorithm
                )));

                if (i == j) {
                    simLabel.setTextFill(Color.GRAY);
                } else if (similarity > 0.7) {
                    simLabel.setTextFill(Color.LIGHTGREEN);
                } else if (similarity > 0.3) {
                    simLabel.setTextFill(Color.YELLOW);
                } else if (similarity > -0.3) {
                    simLabel.setTextFill(Color.LIGHTGRAY);
                } else {
                    simLabel.setTextFill(Color.LIGHTCORAL);
                }

                simLabel.setFont(Font.font("Consolas", 10));
                simLabel.setPrefWidth(55);
                simLabel.setStyle("-fx-background-color: rgba(255,255,255,0.06); -fx-background-radius: 6; -fx-padding: 3 6 3 6;");
                row.getChildren().add(simLabel);
            }
            matrixPanel.getChildren().add(row);
        }

        Text legend = new Text("Légende: Vert=Similaire, Jaune=Moyen, Gris=Faible, Rouge=Opposé");
        legend.setFill(Color.LIGHTGRAY);
        legend.setFont(Font.font("Consolas", 10));
        legend.setWrappingWidth(840);
        matrixPanel.getChildren().add(legend);

        return matrixPanel;
    }

    private void addLinkPredictions(List<String> nodes, Map<String, double[]> projectedEmbeddings,
                                    Map<String, double[]> embeddings, String algorithm,
                                    Pane visualizationPane, double paneWidth, double paneHeight) {
        double centerX = paneWidth / 2;
        double centerY = paneHeight / 2;
        double scale = Math.min(paneWidth, paneHeight) * 0.15;

        int predictionsDrawn = 0;
        int maxPredictions = 8;

        List<String> authors = nodes.stream()
                .filter(node -> graph.getAuthorVertices().contains(node))
                .collect(Collectors.toList());

        List<String> publications = nodes.stream()
                .filter(node -> graph.getPublicationVertices().contains(node))
                .collect(Collectors.toList());

        for (String author : authors) {
            for (String publication : publications) {
                if (predictionsDrawn >= maxPredictions) break;

                if (!graph.getNeighbors(author).contains(publication)) {
                    double similarity = cosineSimilarity(embeddings.get(author), embeddings.get(publication));
                    double probability = (1 + similarity) / 2;

                    double[] proj1 = projectedEmbeddings.get(author);
                    double[] proj2 = projectedEmbeddings.get(publication);

                    if (proj1 != null && proj2 != null) {
                        double x1 = centerX + proj1[0] * scale;
                        double y1 = centerY + proj1[1] * scale;
                        double x2 = centerX + proj2[0] * scale;
                        double y2 = centerY + proj2[1] * scale;

                        Line predictionLine = new Line(x1, y1, x2, y2);
                        predictionLine.setStroke(PREDICTION_COLOR);
                        predictionLine.setStrokeWidth(1.5);
                        predictionLine.getStrokeDashArray().addAll(8.0, 4.0);

                        String tooltipText = String.format(
                                "🔮 Prédiction de lien %s\n%s ↔ %s\nProbabilité: %.3f\nSimilarité: %.4f",
                                algorithm, author, publication, probability, similarity
                        );
                        Tooltip.install(predictionLine, new Tooltip(tooltipText));

                        double midX = (x1 + x2) / 2;
                        double midY = (y1 + y2) / 2;

                        Text probabilityLabel = new Text(String.format("%.3f", probability));
                        probabilityLabel.setFont(Font.font("Consolas", FontWeight.BOLD, 9));
                        probabilityLabel.setFill(Color.LIGHTGREEN);
                        probabilityLabel.setX(midX - 12);
                        probabilityLabel.setY(midY - 5);

                        visualizationPane.getChildren().addAll(predictionLine, probabilityLabel);
                        predictionsDrawn++;
                    }
                }
            }
            if (predictionsDrawn >= maxPredictions) break;
        }
    }

    private void showDetailedSimilarities(String algorithm) {
        Map<String, double[]> embeddings = getEmbeddingsForAlgorithm(algorithm);

        if (embeddings == null || embeddings.isEmpty()) {
            new Alert(Alert.AlertType.WARNING,
                    "Veuillez d'abord calculer les embeddings " + algorithm + "!").show();
            return;
        }

        List<String> analysisNodes = selectRandomNodes(8);

        StringBuilder analysis = new StringBuilder();
        analysis.append("📐 DÉTAIL DES SIMILARITÉS COSINUS - ").append(algorithm).append("\n");
        analysis.append("================================\n\n");

        for (int i = 0; i < analysisNodes.size(); i++) {
            for (int j = i + 1; j < analysisNodes.size(); j++) {
                String node1 = analysisNodes.get(i);
                String node2 = analysisNodes.get(j);
                double similarity = cosineSimilarity(embeddings.get(node1), embeddings.get(node2));
                double angle = Math.acos(Math.max(-1, Math.min(1, similarity))) * 180 / Math.PI;
                double probability = (1 + similarity) / 2;

                analysis.append(String.format("%s ↔ %s\n", node1, node2));
                analysis.append(String.format("  Similarité cosinus: %.4f\n", similarity));
                analysis.append(String.format("  Angle: %.1f°\n", angle));
                analysis.append(String.format("  Probabilité lien: %.2f\n", probability));

                if (similarity > 0.9) analysis.append("  🔥 Très similaires\n");
                else if (similarity > 0.7) analysis.append("  ✅ Similaires\n");
                else if (similarity > 0.3) analysis.append("  ➖ Légèrement similaires\n");
                else if (similarity > -0.3) analysis.append("  ❌ Non similaires\n");
                else analysis.append("  ⚠️  Opposés\n");
                analysis.append("\n");
            }
        }

        TextArea resultArea = new TextArea(analysis.toString());
        resultArea.setEditable(false);
        resultArea.setStyle("-fx-font-family: Consolas; -fx-font-size: 10;");
        resultArea.setPrefSize(600, 450);

        Stage resultStage = new Stage();
        resultStage.setTitle("Détail des Similarités Cosinus - " + algorithm);
        resultStage.setScene(new Scene(new StackPane(resultArea), 620, 470));
        resultStage.show();
    }

    // ==================== UTILITY METHODS ====================

    private double cosineSimilarity(double[] vectorA, double[] vectorB) {
        if (vectorA == null || vectorB == null || vectorA.length != vectorB.length) {
            return -2.0;
        }

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += vectorA[i] * vectorA[i];
            normB += vectorB[i] * vectorB[i];
        }

        if (normA == 0 || normB == 0) return 0.0;

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    private String getShortName(String id, boolean isAuthor) {
        if (isAuthor) {
            String[] parts = id.split("\\s+");
            if (parts.length >= 2) {
                return parts[0].charAt(0) + ". " + parts[parts.length - 1];
            }
            return id.length() > 12 ? id.substring(0, 10) + "…" : id;
        } else {
            return id.length() > 15 ? id.substring(0, 12) + "…" : id;
        }
    }

    private String getShortNameForMatrix(String id, boolean isAuthor) {
        if (isAuthor) {
            String[] parts = id.split("\\s+");
            if (parts.length >= 2) {
                return parts[0].substring(0, 1) + "." + parts[parts.length - 1].substring(0, 3);
            }
            return id.length() > 8 ? id.substring(0, 7) + "…" : id;
        } else {
            String[] words = id.split("\\s+");
            if (words.length >= 2) {
                return words[0].substring(0, 3) + "…" + words[words.length - 1].substring(0, 2);
            }
            return id.length() > 8 ? id.substring(0, 7) + "…" : id;
        }
    }

    private Color getAlgorithmColor(String algorithm) {
        switch (algorithm) {
            case "Node2Vec": return NODE2VEC_COLOR;
            case "DeepWalk": return DEEPWALK_COLOR;
            case "GCN":      return GCN_COLOR;
            case "LINE":     return LINE_COLOR;
            case "GraphSAGE":return GRAPHSAGE_COLOR;
            default: return Color.WHITE;
        }
    }

    private String getAlgoColorHex(String algorithm) {
        switch (algorithm) {
            case "Node2Vec": return "#9b59b6";
            case "DeepWalk": return "#e74c3c";
            case "GCN":      return "#3498db";
            case "LINE":     return "#f39c12";
            case "GraphSAGE":return "#27ae60";
            default: return "#3498db";
        }
    }

    private Button styledButton(String text, String colorHex) {
        Button btn = new Button(text);
        btn.setStyle("-fx-background-color:" + colorHex + "; -fx-text-fill: white; -fx-font-weight: bold; -fx-background-radius: 8; -fx-padding: 6 12;");
        DropShadow shadow = new DropShadow(8, Color.rgb(0,0,0,0.35));
        Glow glow = new Glow(0.0);
        btn.setEffect(shadow);
        btn.setOnMouseEntered(e -> btn.setEffect(glow));
        btn.setOnMouseExited(e -> btn.setEffect(shadow));
        return btn;
    }

    private void updateStats(long layoutTime, int edgesDrawn, long edgeTime) {
        if (graph == null || statsText == null) return;

        StringBuilder stats = new StringBuilder();
        stats.append("Auteurs totaux: ").append(graph.getAuthorVertices().size()).append("\n");
        stats.append("Publications totales: ").append(graph.getPublicationVertices().size()).append("\n");
        stats.append("Arêtes totales: ").append(graph.getEdgeCount()).append("\n");
        stats.append("Auteurs affichés: ").append(Math.min(MAX_DISPLAYED_NODES, graph.getAuthorVertices().size())).append("\n");
        stats.append("Publications affichées: ").append(Math.min(MAX_DISPLAYED_NODES, graph.getPublicationVertices().size())).append("\n");
        stats.append("Auteurs cachés: ").append(Math.max(0, graph.getAuthorVertices().size() - MAX_DISPLAYED_NODES)).append("\n");
        stats.append("Publications cachées: ").append(Math.max(0, graph.getPublicationVertices().size() - MAX_DISPLAYED_NODES)).append("\n\n");

        stats.append("EMBEDDINGS:\n");
        stats.append(node2vecCalculated ? "✅ Node2Vec: " + node2vecEmbeddings.size() + " nœuds\n" : "❌ Node2Vec: Non calculé\n");
        stats.append(deepwalkCalculated ? "✅ DeepWalk: " + deepwalkEmbeddings.size() + " nœuds\n" : "❌ DeepWalk: Non calculé\n");
        stats.append(gcnCalculated ? "✅ GCN: " + gcnEmbeddings.size() + " nœuds\n" : "❌ GCN: Non calculé\n");
        stats.append(lineCalculated ? "✅ LINE: " + lineEmbeddings.size() + " nœuds\n" : "❌ LINE: Non calculé\n");
        stats.append(graphSageCalculated ? "✅ GraphSAGE: " + graphSageEmbeddings.size() + " nœuds\n" : "❌ GraphSAGE: Non calculé\n");
        stats.append("Dimensions: 64\n\n");

        if (layoutTime > 0) stats.append("Temps layout: ").append(layoutTime).append("ms\n");

        if (edgesDrawn > 0) {
            stats.append("Arêtes visibles: ").append(edgesDrawn).append("\n");
            stats.append("Temps dessin: ").append(edgeTime).append("ms");
        } else {
            stats.append("Arêtes: Cliquer 'Afficher'");
        }

        statsText.setText(stats.toString());
    }

    // Progress dialog class
    private static class ProgressDialog extends Stage {
        private ProgressBar progressBar;
        private Label progressLabel;

        public ProgressDialog(String title) {
            setTitle(title);
            setWidth(420);
            setHeight(160);
            setResizable(false);

            VBox content = new VBox(20);
            content.setAlignment(Pos.CENTER);
            content.setPadding(new Insets(20));
            content.setStyle("-fx-background-color: #2c3e50; -fx-background-radius: 8;");

            progressLabel = new Label("Initialisation...");
            progressLabel.setTextFill(Color.WHITE);

            progressBar = new ProgressBar(0);
            progressBar.setPrefWidth(350);

            content.getChildren().addAll(progressLabel, progressBar);
            Scene scene = new Scene(content);
            setScene(scene);
        }

        public void updateProgress(String message, double progress) {
            progressLabel.setText(message);
            progressBar.setProgress(progress);
        }
    }

    // Prediction class
    private static class Prediction {
        String node1, node2;
        double probability, similarity;

        Prediction(String node1, String node2, double probability, double similarity) {
            this.node1 = node1;
            this.node2 = node2;
            this.probability = probability;
            this.similarity = similarity;
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}