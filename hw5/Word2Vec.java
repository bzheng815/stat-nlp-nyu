package nlp.assignments;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.*;

import nlp.assignments.LanguageModelTester.SentenceCollection;
import nlp.math.HuffmanCoding;
import nlp.math.HuffmanCoding.HuffmanNode;
import nlp.util.Counter;
import nlp.util.CommandLineUtils;

public class Word2Vec{

    
    public static double[] neu;
    public static double[] neu_e;
    public static double[][] inputWeight;
    public static double[][] outputWeight;
    public static double[][] negWeight;
    
    public static int MAX_16RANDOM = 65535;
    
    public static Counter<String> vocab = new Counter<String>();
    public static Map<String, HuffmanNode> huffmanNodes;
    public static int vocabSize;


    public static void GetVocab(String dataPath){
       
        Collection<List<String>> sentenceCollection = SentenceCollection.Reader
            .readSentenceCollection(dataPath);
        
        for (List<String> sentence : sentenceCollection) {
            for (String word : sentence) {
                vocab.incrementCount(word, 1.0);
            } 
        }
    }
    
    
    public static void initInWeight(int dimension) {
        inputWeight = new double[vocabSize][dimension];
        Random r = new Random();
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < dimension; j++) {
                long rd = r.nextInt(MAX_16RANDOM);
                inputWeight[i][j] = (rd/(double)MAX_16RANDOM)/dimension; 
            }
        }
    }
    
    //Neural Network Training
    public static HashMap<String, float[]> TrainEmbeddings(String dataPath, 
        int dimension, 
        int window, 
        double alpha, 
        int negativeSamples,
        int iter){
   
        GetVocab(dataPath);
        huffmanNodes = new HuffmanCoding(vocab).encode();
        vocabSize = huffmanNodes.size();
        initInWeight(dimension);
         
        neu = new double[dimension];
        neu_e = new double[dimension];
        outputWeight = new double[vocabSize][dimension];
        negWeight = new double[vocabSize][dimension];

        Collection<List<String>> sentenceCollection = SentenceCollection.Reader
            .readSentenceCollection(dataPath);

        for (int i = 0; i< iter; i++){
            for (List<String> sentence : sentenceCollection) {
            //System.out.println(sentence);
                CBOWSentenceTrain(sentence, dimension, window, alpha, negativeSamples);
            }
        }
        
        HashMap<String, float[]> embeddings = new HashMap<String, float[]>();
        
        for (String word : vocab.keySet()) {
            //System.out.println(word);
            embeddings.put(word, new float[dimension]);
            int idx = huffmanNodes.get(word).idx;
            for (int j = 0; j < dimension; j++) {
                embeddings.get(word)[j] = (float) inputWeight[idx][j];
            }
        }
        
        return embeddings;
    }
    
    
    
    //CBOW Train Sentence
    public static void CBOWSentenceTrain(List<String> sentence, int dimension, int window, double alpha, int negativeSamples){
        
        //System.out.println(sentence);
        int sentenceLength = sentence.size();
    
        Random r = new Random();
        for (int sentencePosition = 0; sentencePosition < sentenceLength; sentencePosition++) {
            String word = sentence.get(sentencePosition);
            HuffmanNode huffmanNode = huffmanNodes.get(word);
            for (int j = 0; j < dimension; j++){
                neu[j] = 0;
                neu_e[j] = 0;
            }
            int start = r.nextInt(window);

            // in -> hidden
            int c = 0;
            for (int i = start; i < window * 2 + 1 -start; i++) {
                if (i == window)
                    continue;
                int wordPosition = sentencePosition - window + i;
                if (wordPosition < 0 || wordPosition >= sentenceLength)
                    continue;
                int idx = huffmanNodes.get(sentence.get(wordPosition)).idx;
                for (int j = 0; j < dimension; j++)
                    neu[j] += inputWeight[idx][j];
                c++;
            }

            if (c == 0)
                continue;
            for (int j = 0; j < dimension; j++)
                neu[j] /= c;
    
            //useHierarchicalSoftmax
            
            boolean useHierarchicalSoftmax = true;
            if (useHierarchicalSoftmax) {
                for (int i = 0; i < huffmanNode.code.length; i++) {
                    double f = 0;
                    int point = huffmanNode.point[i];
                    
                    for (int j = 0; j < dimension; j++) 
                        f += neu[j] * outputWeight[point][j];
                    
                    double EXP = Math.exp(f);
                    double g = (1 - huffmanNode.code[i] - (15*EXP)/(1+15*EXP)) * alpha;
                    
                    for (int j = 0; j < dimension; j++)
                        neu_e[j] += g * outputWeight[point][j];
                    for (int j = 0; j < dimension; j++)
                        outputWeight[point][j] += g * neu[j];
                }
            }


            //handleNegativeSampling
            for (int i = 0; i <= negativeSamples; i++) {
                int point;
                if (i == 0) {
                    point = huffmanNode.idx;
                }
                else {
                    Random rr = new Random();
                    point = rr.nextInt(vocabSize-2)+1;
                }
                
                double f = 0;
                for (int j = 0; j< dimension; j++)
                    f += neu[j] * negWeight[point][j];
                
                double EXP = Math.exp(f);
                double g = (1 - (15*EXP)/(1+15*EXP)) * alpha;
                
                for (int j = 0; j< dimension; j++)
                    neu_e[j] += g * negWeight[point][j];
                for (int j = 0; j< dimension; j++)
                    negWeight[point][j] += g * neu[j];
            }
            
            // hidden -> in
            for (int i = start; i < window * 2 + 1 -start; i++) {
                if (i == window)
                    continue;
                int wordPosition = sentencePosition - window + i;
                if (wordPosition < 0 || wordPosition >= sentenceLength)
                    continue;
                int idx = huffmanNodes.get(sentence.get(wordPosition)).idx;
                for (int j = 0; j < dimension; j++)
                    inputWeight[idx][j] += neu_e[j];
            }
  
        }
        
    }
  
 
    private static void writeEmbeddings (HashMap<String, float[]> embeddings,
            String path, int embeddingDim) throws Exception {
        BufferedWriter writer = new BufferedWriter(new FileWriter(path));
        writer.write(embeddings.size() + " " + embeddingDim + "\n");
        for (Map.Entry<String, float[]> wordEmbedding : embeddings.entrySet()) {
            String word = wordEmbedding.getKey();
            String embeddingString = Arrays.toString(wordEmbedding.getValue())
                .replace(", ", " ").replace("[", "").replace("]", "");
            if (wordEmbedding.getValue().length != embeddingDim) {
                System.out.println("The embedding for " + word + " is not "
                        + embeddingDim + "D.");
                System.exit(0);
            }
            writer.write(word + " " + embeddingString + "\n");
        }
        writer.close();
    }

    public static void main(String[] args) throws Exception {
        // Parse command line flags and arguments.
        Map<String, String> argMap = CommandLineUtils
            .simpleCommandLineParser(args);
        
        // Read commandline parameters.
        String dataPath = "";
        if (!argMap.containsKey("-trainingdata")) {
            System.out.println("-training data required");
            System.exit(0);
        }
        else {
            dataPath = argMap.get("-trainingdata");
        }
        String embeddingPath = "";
        if (!argMap.containsKey("-output")) {
            System.out.println("-output path required");
            System.exit(0);
        }
        else{
            embeddingPath = argMap.get("-output");
        }
        int dimension = 0;
        if (!argMap.containsKey("-size")) {
            System.out.println("-layer size required");
            System.exit(0);
        }
        else{
            dimension = Integer.parseInt(argMap.get("-size"));
        }
        int window = 0;
        if (!argMap.containsKey("-window")) {
            System.out.println("-window size required");
            System.exit(0);
        }
        else {
            window = Integer.parseInt(argMap.get("-window"));
        }
        double alpha = 0.0;
        if (!argMap.containsKey("-alpha")) {
            System.out.println("-learning rate required");
            System.exit(0);
        }
        else{
            alpha = Double.parseDouble(argMap.get("-alpha"));
        }
        int negativeSamples = 0;
        if (!argMap.containsKey("-negative")) {
            System.out.println("-negative samples required");
            System.exit(0);
        }
        else{
            negativeSamples = Integer.parseInt(argMap.get("-negative"));
        }
        int iter = 0;
        if (!argMap.containsKey("-iter")) {
            System.out.println("-iterations required");
            System.exit(0);
        }
        else{
            iter = Integer.parseInt(argMap.get("-iter"));
        }
        HashMap<String, float[]> embeddings = 
            TrainEmbeddings(dataPath, dimension, window, alpha, negativeSamples, iter);
        
        writeEmbeddings(embeddings, embeddingPath, dimension);
    }


}
