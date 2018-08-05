import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Random;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.core.AttributeStats;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;

class BasicRule 
{
    int attributeIndex;
	int attributeValue;
	
	public BasicRule()
	{
			
	}
		
	/**
	 * Instantiates a new basic rule.
	 *
	 * @param attributeIndex the attribute index
	 * @param attributeValue the attribute value
	 */
	public BasicRule(int attributeIndex, int attributeValue)
	{
		this.attributeIndex = attributeIndex;
		this.attributeValue = attributeValue;
	}
}

class Rule 
{
   	List<BasicRule> basicRule;
   	double returnValue;		// class value.
   	
   	/**
	    * Instantiates a new rule.
	    */
	   public Rule()
   	{
   		basicRule = new LinkedList<BasicRule>();
   	}
}

class Node 
{
	LinkedList<Node> children;
	Node parent;
	int attributeIndex;
	double returnValue;
   	Rule nodeRule = new Rule();
   	Instances trainingExamples;

   	/**
   	 * Instantiates a new node.
   	 *
   	 * @param parent - node's parent in the tree
   	 * @param trainingExamples the relevant training examples for the node
   	 */
   	public Node(Node parent, Instances trainingExamples)
   	{
   		this.parent=parent;
   		this.trainingExamples=trainingExamples;
   		this.children = new LinkedList<Node>();
   	}
   	
   	// Empty constructor
   	public Node()
   	{
   		this.children = new LinkedList<Node>();
   	}
}

/**
 * The Class DecisionTree.
 */
public class DecisionTree implements Classifier 
{

	/** The root node. */
	private Node rootNode;

	/**
	 * The Enum PruningMode.
	 */
	public enum PruningMode 
	{
		/** The None. */
		None, 
		/** The Chi. */
		Chi, 
		/** The Rule. */
		Rule
	};

		/** The m pruning mode. */
		private PruningMode m_pruningMode;

	   /** The validation set. */
	   Instances validationSet;
   	
	   /** The rules. */
	   private List<Rule> rules;
   	
	   /** The queue. */
	   // Initialize a queue to hold the nodes at.
   	LinkedList<Node> queue;
   	
   	/**
	    * Instantiates a new decision tree.
	    *
	    * @param trainingData the training data
	    */
	   public DecisionTree(Instances trainingData, Instances validationData)
   	{
		queue = new LinkedList<Node>();
   		rules = new ArrayList<Rule>();
   		// Initialize root node.
   		this.rootNode = new Node();
   		this.rootNode.parent=null;
   		this.rootNode.trainingExamples = new Instances(trainingData);
   		this.validationSet = validationData;
   	} 	
   	
	/**
	 * Gets the rules.
	 *
	 * @return the rules
	 */
	public List<Rule> getRules() 
	{
		return rules;
	}

	/**
	 * Builds the classifier.
	 *
	 * @param arg0 the set of instances - train the tree accordingly to arg0.
	 * @throws Exception.
	 */
	@Override
	public void buildClassifier(Instances arg0) throws Exception 
	{
		queue.add(this.rootNode);
		buildTree(arg0);
		converTreeToRules();
	}

	/**
	 * Gets the all leaf nodes.
	 *
	 * @param node the node
	 * @return the all leaf nodes
	 */
	private Set<Node> getAllLeafNodes(Node node) 
	{
	    Set<Node> leafNodes = new HashSet<Node>();
	    if (node.children.isEmpty()) 
	    {
	        leafNodes.add(node);
	    }
	    else
	    {
	        for (Node child : node.children) 
	        {
	            leafNodes.addAll(getAllLeafNodes(child));
	        }
	    }
	    return leafNodes;
	}
	
	/**
	 * Convert tree to rules.
	 */
	private void converTreeToRules() 
	{
		Node nodeToIterateOn;
		Set<Node> leaves = getAllLeafNodes(this.rootNode);
		for (Node leaf : leaves)
		{
			// traverse the tree to the root and construct a Rule made out of simple rules.
			nodeToIterateOn = leaf;
			// create a rule to iterate on and then insert it to the tree's rules field.
			Rule newRule = new Rule();
			while(nodeToIterateOn.parent!=null)
			{
				// extract the BasicRule's attribute index from its father.
				int attributeIndex = nodeToIterateOn.parent.attributeIndex;
				// extract the BasicRule's attribute's value from its father. (all instances have same classification in a leaf)
				int attributeValue = (int) nodeToIterateOn.trainingExamples.get(0).value(attributeIndex);
				// adding basicRules in the opposite direction. (bottom of the tree to its top)
				newRule.basicRule.add(new BasicRule(attributeIndex, attributeValue));
				// traverse up the tree.
				nodeToIterateOn = nodeToIterateOn.parent;
			}
			
			// reverse the tree. ( adding basicRules in opposite direction, needed to reverse)
			Collections.reverse(newRule.basicRule);
			newRule.returnValue = leaf.returnValue;
			this.rules.add(newRule);
			newRule = new Rule();
		}
		
		if(this.m_pruningMode == PruningMode.Rule)
		{
			rulePruning();
		}
	}

	/**
	 * Sets the pruning mode.
	 *
	 * @param pruningMode the new pruning mode
	 */
	public void setPruningMode(PruningMode pruningMode) {
		m_pruningMode = pruningMode;
	}
	
	/**
	 * Sets the validation.
	 *
	 * @param validation the new validation
	 */
	public void setValidation(Instances validation) {
		validationSet = validation;
	}

	/**
	 * Calculate the average on a given instances set (could be the training, test or validation set).
	 * The average error is the total number of classification mistakes on the input instances set and
	 * divides that by the number of instances in the input set.
	 *
	 * @param instancesSet - Instances object
	 * @return double - Average error
	 */
	public double calcAvgError(Instances instancesSet)
	{
		int mistakes = 0;
		for(int i=0;i<instancesSet.numInstances();i++)
		{
			if(instancesSet.get(i).value(instancesSet.classIndex())!= classifyInstance(instancesSet.get(i)))
			{
				mistakes++;
			}
		}
		return (double) mistakes / (double) instancesSet.numInstances();
	}

	/**
	 * Builds the tree.
	 *
	 * @param data_instances the data instances to use when constructing the tree.
	 */
	private void buildTree(Instances data_instances)
	{
		int bestDesicionAttribute;
		Node nodeToIterateOn;	
		// Iterate over all nodes in the queue.
		while(!queue.isEmpty())
		{
			nodeToIterateOn = queue.removeFirst();
			switch(nodeToIterateOn.trainingExamples.numInstances()) 
			{
			   case 0 :
			      break;
			   case 1 :
				   nodeToIterateOn.returnValue=nodeToIterateOn.trainingExamples.get(0).classValue();
				   break;
			   case 2 :
				   nodeToIterateOn.returnValue=nodeToIterateOn.trainingExamples.get(0).classValue();
				   break; 
			   default : 
			   {
				   bestDesicionAttribute = chooseBestDesicionAttribute(nodeToIterateOn.trainingExamples);
					if(bestDesicionAttribute != -1)
					{
						 if( this.m_pruningMode == PruningMode.Chi && calcChiSquare(nodeToIterateOn.trainingExamples, bestDesicionAttribute) < 15.51)
						 {
							 // need to prune accordingly to Chi Square statistics.
							 nodeToIterateOn.returnValue = getMajorityReturnValue(nodeToIterateOn);
						 }
						 else
						 {
							 nodeToIterateOn.attributeIndex=bestDesicionAttribute;
							 addDescendentToQueue(nodeToIterateOn, nodeToIterateOn.attributeIndex);
						 }
					}
					else
					{
						if(calcEntropy(nodeToIterateOn.trainingExamples) == 0)
							nodeToIterateOn.returnValue = nodeToIterateOn.trainingExamples.get(0).classValue();
						else
						{
							nodeToIterateOn.returnValue = getMajorityReturnValue(nodeToIterateOn);
						}
					}
			   }
			}
		}
	}

	/**
	 * Gets the return value accordingly to the majority of return values of the instances within nodeToIterateOn
	 *
	 * @param nodeToIterateOn - contains the instances needed to be checked
	 * @return classification, 1 or 0.
	 */

	private int getMajorityReturnValue(Node nodeToIterateOn) 
	{
		int [] classCounts = new int[nodeToIterateOn.trainingExamples.numClasses()];
		classCounts[0]=0;
		classCounts[1]=0;
		for( int i=0;i<nodeToIterateOn.trainingExamples.numInstances();i++)
		{
			if(nodeToIterateOn.trainingExamples.get(i).classValue() == 0)
			{
				classCounts[0]++;
			}
			else
			{
				classCounts[1]++;
			}
		}
		if(classCounts[0] > classCounts[1])
		{
			return 0;
		}
		else
		{
			return 1;
		}
	}

	/**
	 * Adds the descendant nodes to queue.
	 *
	 * @param father the father node
	 * @param attributeIndex the attribute index
	 */
	private void addDescendentToQueue(Node father,int attributeIndex)
	{	
		Node newNodeToAdd;
		// Descendants - an array containing newly created nodes whose instances corresponds to a different value of attributeIndex.
		Instances[] descendents = new Instances[father.trainingExamples.numDistinctValues(attributeIndex)];
		// Initialize the descendants array.
		for(int i=0;i<father.trainingExamples.numDistinctValues(attributeIndex);i++)
		{
			descendents[i] = new Instances(father.trainingExamples, 0, 0);
		}
		// sort the instances in respect to attributeIndex, in order to distinct them to "bins" by their attributeIndex's value.
		father.trainingExamples.sort(attributeIndex);
		// Maintain the lastValueIteratedOn to distinguish between different values to place them at different cells.
		double lastValueIteratedOn = father.trainingExamples.get(0).value(attributeIndex);
		// descendentsValueIterator - Since instances are sorted we iterate them accordingly to their value and thus need to keep track of current "bin".
		int descendentsValueIterator = 0;
		for(int i=0;i<father.trainingExamples.numInstances();i++)
		{
			if(father.trainingExamples.get(i).value(attributeIndex) == lastValueIteratedOn)
			{
				descendents[descendentsValueIterator].add(father.trainingExamples.get(i));
			}
			else
			{
				// Move to the next value to place the instances corresponding to it in the right "bin".
				descendentsValueIterator++;
				lastValueIteratedOn=father.trainingExamples.get(i).value(attributeIndex);
				descendents[descendentsValueIterator].add(father.trainingExamples.get(i));
			}
		}
		// Iterate over the bins and create a node out of every Instances object to place in queue.
		for(int i=0;i<father.trainingExamples.numDistinctValues(attributeIndex);i++)
		{
			if(descendents[i].size()!=0)
			{
				newNodeToAdd = new Node(father,descendents[i]);
				newNodeToAdd.parent = father;
				this.queue.addLast(newNodeToAdd);
				// update the father's children's' field.
				father.children.add(newNodeToAdd);
			}
		}
	}
	
		 
    /**
     * Choose the best decision attribute.
     *
     * @param trainingExamples - the training examples
     * @return the index of the best decision attribute or -1 if all instances have identical values for all attributes but classIndex.
     */
    
    private int chooseBestDesicionAttribute(Instances trainingExamples) 
    {
    	int maxImpurityReduceAttribute=-1;
    	double maxImpurityReduceValue = 0;
    	// Compare the information gain double for each attribute and choose the one who maximizes the gain.
    	for(int i=0;i<trainingExamples.numAttributes()-1;i++)
    	{
    		if(calcInfoGain(trainingExamples,i) > maxImpurityReduceValue)
    		{
    			maxImpurityReduceAttribute = i;
    			maxImpurityReduceValue = calcInfoGain(trainingExamples, i);
    		}
    	}
    	return maxImpurityReduceAttribute;
    }
    
	/**
	 * find all rules which can be pruned off the list of rules and remove them.
	 */
	public void rulePruning() 
	{
		double errorBeforePruning;
		int ruleToPrune;
		while( (ruleToPrune = findBestRuleToPrune() ) != -1)
		{
			errorBeforePruning = calcAvgError(validationSet);
			Rule temp = rules.get(ruleToPrune);
			rules.remove(ruleToPrune);
			if(errorBeforePruning < calcAvgError(validationSet))
			{
				// The removal of the rule hasn't improved the error.
				rules.add(temp);
				return;
			}
		}
	}


	/**
	 * Finds the best rule to prune.
	 *
	 * @return the index of in the set of Rules.
	 */
	public int findBestRuleToPrune() 
	{
		double errorAfterPruning;
		Rule currentRuleToIterateOn;
		double minimumError = calcAvgError(validationSet);
		int bestRule_index = -1;
		for(int i=0; i<rules.size(); i++) 
		{
			currentRuleToIterateOn = rules.get(i);
			rules.remove(i);
			// check the error after removing each rule.
			errorAfterPruning = calcAvgError(validationSet);
			if(errorAfterPruning < minimumError) 
			{
				minimumError = errorAfterPruning;
				bestRule_index = i;
			}
			// Add back the rule to test the minimum error of removing the next rule in Rules.
			rules.add(i, currentRuleToIterateOn);
		}
		return bestRule_index; 
	}

	/**
	 * Calculate Chi square statistics
	 *
	 * @param instances - the set of instances of which we decide whether to prune the node in the tree or not
	 * @param attributeIndex - the index of the attribute which is being checked.
	 * @return Chi square statistics accordingly to the formula.
	 */
	public double calcChiSquare(Instances instances, int attributeIndex) 
	{
		double e0,e1;
		int positivesNumber=0,negativesNumber=0;
		Instances[] instancesValuesByattributeIndex = splitInstances(instances, attributeIndex);
		double chiSquareStatistics = 0;
		double probability_y0 = 0,probability_y1 = 0;
		for(int i=0; i<instances.numInstances(); i++) 
		{
			if(instances.get(i).classValue() == 1.0)
			{
				probability_y1++;
			}
			else 
			{
				probability_y0++;
			}
		}
		probability_y0 /= instances.numInstances();
		probability_y1 /= instances.numInstances();
		for(int i=0; i<instancesValuesByattributeIndex.length; i++) 
		{
			positivesNumber=0;
			negativesNumber=0;
			for(int j=0; j<instancesValuesByattributeIndex[i].numInstances(); j++) 
			{
				if(instancesValuesByattributeIndex[i].get(j).classValue() == 1.0)  
					negativesNumber++; 
				else 
					positivesNumber++;
			}
			if(instancesValuesByattributeIndex[i].numInstances()>0)
			{
				e0 = instancesValuesByattributeIndex[i].numInstances() * probability_y1;
				e1 = instancesValuesByattributeIndex[i].numInstances() * probability_y0;
				chiSquareStatistics += ( ( Math.pow( ( positivesNumber - e0 ) , 2) / e0 ) + ( Math.pow( (negativesNumber - e1 ) , 2 ) / e1));
			}
		}	
		return chiSquareStatistics;
	}

	
	/**
     * Calculates the entropy of a given instances.
     *
     * @param instances - the instances.
     * @return Returns a double representing the entropy value.
     */

	private double calcEntropy(Instances instances)
	{
		int [] classCounts = new int[instances.numClasses()];
		double entropy = 0;
		for( int i=0;i<instances.numInstances();i++)
		{
			if(instances.get(i).classValue() == 1.0)
				classCounts[1]++;
			else
				classCounts[0]++;
		}
		for (int i = 0; i < classCounts.length; i++) 
		{
			if (classCounts[i] != 0) 
			{
				entropy -= ( (double) classCounts[i] / (double) instances.numInstances() ) * ( Math.log(( (double) classCounts[i] / (double) instances.numInstances() )) / Math.log(2));
			}
			else
				return 0;
		}
		return entropy;
	}


	/**
	 *  calculates the information gain of splitting 
	 *  the input data according to the attribute.
	 * 
	 * @param instances - Instances object (a subset of the training data).
	 * @param attributeIndex - attribute index (int).
	 * @return infoGain - The information gain (double). 
	 */
	
	private double calcInfoGain(Instances instances, int attributeIndex)
	{	
		AttributeStats tempAttributeStats = instances.attributeStats(attributeIndex);
		int numberOfValues = tempAttributeStats.totalCount;
		double infoGain = calcEntropy(instances);
		Instances[] splitData = splitInstances(instances, attributeIndex);
		for (int i = 0; i < numberOfValues; i++) 
		{
			if (splitData[i].numInstances() > 0) 
			{
				infoGain -= ((double)(splitData[i].numInstances() / instances.numInstances())) * calcEntropy(splitData[i]);
			}
		}
		return infoGain;
	}
	
	/**
	 * Split the instances to different array cells accordingly the value of  "attributeIndex" in each instance in "instances".
	 *
	 * @param instances - the set of instances to split the instances from.
	 * @param attributeIndex - the attribute of the index of which we split instances accordingly to.
	 * @return array of instances objects, sorted by their value of attribute number attributeIndex.
	 */
	public Instances[] splitInstances(Instances instances, int attributeIndex) 
	{
		double lastVal;
		AttributeStats tempAttributeStats = instances.attributeStats(attributeIndex);
		int numberOfValues = tempAttributeStats.totalCount;
		// sort the instances by the value of attributeIndex to traverse them and save them to attributeValues array by their value.
		instances.sort(attributeIndex);
		// attributeValues - size: number of values accordingly to attributeIndex
		Instances[] attributeValues = new Instances[numberOfValues];
		for(int i = 0; i < attributeValues.length; i++) 
		{
			//Initialize each cell of attributeValues array.
			attributeValues[i] = new Instances(instances, instances.numInstances());
		}
		// save lastVal as the last value we iterated on with respect to the instances (we sorted it before hand).
		lastVal = instances.get(0).value(attributeIndex); 
		for(int i=0,j=0; i<instances.numInstances(); i++) 
		{
			if(instances.get(i).value(attributeIndex) != lastVal) 
			{
				lastVal = instances.get(i).value(attributeIndex);
				j++;
			}
			attributeValues[j].add(instances.get(i));
		}

		return attributeValues;
	}
	
	/**
	 * Classifies a new instance.
	 *
	 * @param instance the instance
	 * @return the double
	 */
	@Override
	public double classifyInstance(Instance instance) 
	{
		int bestRuleValue = 0;
		int basicRules_number;
		Rule bestRule = new Rule();
		// Iterate over each of the rules to find the best one.
		for(Rule rule : rules) 
		{
			// reset the number of basic rules each iteration of a new rule.
			basicRules_number = 0;
			for(int i=0; i<rule.basicRule.size(); i++) 
			{
				if(instance.value(rule.basicRule.get(i).attributeIndex) == rule.basicRule.get(i).attributeValue) 
				{
					basicRules_number++;
				}
				else 
				{
					break;
				}
			}
			if(basicRules_number > bestRuleValue) 
			{
				bestRule = rule;
				bestRuleValue = basicRules_number;
			}
		}
		return bestRule.returnValue;
	}
    
	
    /**
     * Distribution for instance.
     *
     * @param arg0 the arg 0
     * @return the double[]
     * @throws Exception the exception
     */
    @Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	/**
	 * Gets the capabilities.
	 *
	 * @return the capabilities
	 */
	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}

}
