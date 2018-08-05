import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class MainHW2 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
	
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	

	public static void main(String[] args) throws Exception {
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");
		
		// Build the tree without pruning.
		DecisionTree bTree_withoutPruning = new DecisionTree(trainingCancer , validationCancer);
		// Build the tree out of the TRAINING data.
		bTree_withoutPruning.buildClassifier(trainingCancer);
		System.out.println("Decision Tree with No pruning");
        System.out.println("The average train error of the decision tree is "+ bTree_withoutPruning.calcAvgError(trainingCancer));
        System.out.println("The average test error of the decision tree is "+bTree_withoutPruning.calcAvgError(testingCancer));
        System.out.println("The amount of rules generated from the tree "+bTree_withoutPruning.getRules().size());
        
        System.out.println("Decision Tree with Chi pruning");
        // build the tree with pruning option set to ChiSquare.
        DecisionTree bTree_chiSquare = new DecisionTree(trainingCancer, validationCancer);
        // Set pruning mode to chiSquare.
        bTree_chiSquare.setPruningMode(DecisionTree.PruningMode.Chi);
		// Build the tree out of the TRAINING data.
        bTree_chiSquare.buildClassifier(trainingCancer);
        System.out.println("The average train error of the decision tree is "+ bTree_chiSquare.calcAvgError(trainingCancer));
        System.out.println("The average test error of the decision tree is "+bTree_chiSquare.calcAvgError(testingCancer));
        System.out.println("The amount of rules generated from the tree "+bTree_chiSquare.getRules().size());
        
        System.out.println("Decision Tree with Rule pruning");
        // build the tree with pruning option set to ChiSquare.
        DecisionTree bTree_rulePruning = new DecisionTree(trainingCancer, validationCancer);
        // Set pruning mode to chiSquare.
        bTree_rulePruning.setPruningMode(DecisionTree.PruningMode.Rule);
		// Build the tree out of the TRAINING data.
        bTree_rulePruning.buildClassifier(trainingCancer);
        System.out.println("The average train error of the decision tree is "+ bTree_rulePruning.calcAvgError(trainingCancer));
        System.out.println("The average test error of the decision tree is "+bTree_rulePruning.calcAvgError(testingCancer));
        System.out.println("The amount of rules generated from the tree "+bTree_rulePruning.getRules().size());
	}
}
