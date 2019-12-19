import pickle
import json
import lime
import scipy
from lime.lime_text import LimeTextExplainer
#dictionary [key, explanation]

class tuner:
    def __init__(self, text, filename, class_names):
        self.dict = {}
        self.sigmas = []
        self.text = text
        self.filename = filename
        self.class_names = class_names
    
    def initialize():
        self.dict.clear()
        return
    
    #get the blackbox prediction for a text instance using filename(model.pkl file) 
    def get_Prediction(self):
        print("HII")
        loaded_model = pickle.load(open(self.filename, 'rb'))
        result = loaded_model.predict_proba([self.text])
        return json.dumps(result.tolist())
    
    #get the explanations or fill up all the explanantion field for a particular sigma in dictionary with key value as sigma
    def get_Explanation(self, sigma):
        if sigma in self.dict:
            print("YES")
            return self.dict[sigma]
        else:
            explainer = LimeTextExplainer(kernel_width=sigma,class_names=self.class_names)
            print("sigma = " + str(sigma))
            black_box = pickle.load(open(self.filename, 'rb'))
            exp = explainer.explain_instance(self.text, black_box.predict_proba, num_features=6)
            self.dict[sigma] = exp
            return self.dict[sigma]
        return 
    
    #to plot the entropy v/s sigma curve, this fuction calulates the value of entropy for all the sigmas and then return sigmas, entropies
    def get_Sigma_Entropy(self):
        sigmas = []
        entropies = []
        for key in sorted (self.dict.keys()):
            weight_per_bucket = []
            for b in range(11):
                weight_per_bucket.append(0)
            exp = self.dict[key]
            list_pair = [(p1, p2) for idx1, p1 in enumerate(exp.weights) for idx2, p2 in enumerate(exp.bb_labels) if idx1==idx2]
            list_pair.sort()
            for t in list_pair:
                if(t[1] >= 0 and t[1] < 0.1):
                    weight_per_bucket[0] += t[0]
                elif(t[1] >= 0.1 and t[1] < 0.2):
                    weight_per_bucket[1] += t[0]
                elif(t[1] >= 0.2 and t[1] < 0.3):
                    weight_per_bucket[2] += t[0]
                elif(t[1] >= 0.3 and t[1] < 0.4):
                    weight_per_bucket[3] += t[0]
                elif(t[1] >= 0.4 and t[1] < 0.5):
                    weight_per_bucket[4] += t[0]
                elif(t[1] >= 0.5 and t[1] < 0.6):
                    weight_per_bucket[5] += t[0]
                elif(t[1] >= 0.6 and t[1] < 0.7):
                    weight_per_bucket[6] += t[0]
                elif(t[1] >= 0.7 and t[1] < 0.8):
                    weight_per_bucket[7] += t[0]
                elif(t[1] >= 0.8 and t[1] < 0.9):
                    weight_per_bucket[8] += t[0]
                elif(t[1] >= 0.9 and t[1] < 1.0):
                    weight_per_bucket[9] += t[0]
                else :
                    weight_per_bucket[10] += t[0]
            norm_weight_per_bucket = [float(i)/sum(weight_per_bucket) for i in weight_per_bucket]
            entropy = scipy.stats.entropy(norm_weight_per_bucket)
            self.dict[key].entropy = entropy
            sigmas.append(key)
            entropies.append(entropy)
        return sigmas, entropies
      
    #to be called only after the get_Explanation for sigma is called
    #used to plot the weight distribution plot of neighbourhood points (vary for each sigma)
    def get_Weight_Distribution_Plot_Per_Sigma(self, sigma):
        return self.dict[sigma].weights, self.dict[sigma].bb_labels
    
    #to be called only after the get_Explanation for sigma is called
    #returns all the 4 models' rmse (constant model, linear model(used by LIME), decision tree, random forests)    
    def get_RMSEs_And_Sigmas(self):
        constant_rmse = []
        linear_rmse = []
        decisionTree_rmse = []
        randomForest_rmse = []
        sigmas = []
        for key in sorted (self.dict.keys()):
            sigmas.append(key)
            constant_rmse.append(self.dict[key].naive_score)
            linear_rmse.append(self.dict[key].score)
            decisionTree_rmse.append(self.dict[key].rf_score)
            randomForest_rmse.append(self.dict[key].rf_2_score)
        return sigmas, constant_rmse, linear_rmse, decisionTree_rmse, randomForest_rmse
    
    def getRange_Constant_Model(self):
        sigmas,constant_rmse, linear_rmse, decisionTree_rmse, randomForest_rmse = self.get_RMSEs_And_Sigmas()
        const_range = []
        const_range.append(0.39)
        for i in range(len(sigmas)):
            if(linear_rmse[i] < constant_rmse[i]):
                const_range.append(sigmas[i])
                return const_range
        const_range.append(0.39)
        return const_range
    
    def getRange_Linear_Model(self):
        sigmas,constant_rmse, linear_rmse, decisionTree_rmse, randomForest_rmse = self.get_RMSEs_And_Sigmas()
        const_range = self.getRange_Constant_Model()
        linear_range = []
        linear_range.append(const_range[1])
        diff = [m - n for m,n in zip(linear_rmse,randomForest_rmse)]
        for i in range(len(sigmas)):
            if(diff[len(sigmas)-i-1] < 0):
                linear_range.append(sigmas[len(sigmas)-i-1])
                return linear_range
        linear_range.append(const_range[1])
        return linear_range
        
            