#include <deque>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define TRAIN_DATA "train.csv"
#define TEST_DATA "test.csv"
#define OUTPUT_DATA "RF_result.csv"
#define MODEL_DATA "RF.model"

#define NUM_TREES 200
#define NUM_THREADS 10
#define M_ATTRS 50
#define NUM_SPLIT_ENUM 20
#define MIN_NODE_SIZE 10
#define TREE_DEPTH 20

using namespace std;

struct CARTNode {
    CARTNode(int at, float val, CARTNode *l = NULL, CARTNode *r = NULL) {
        attr = at, value = val, left = l, right = r;
    }

    int attr;
    float value;
    CARTNode *left, *right;
};

struct Interval {
    Interval(int lv = -1, int f = -1, int t = -1, int pv = -1, CARTNode *p = NULL) {
        level = lv, from = f, to = t, parentValue = pv, parent = p;
    }

    int level, from, to, parentValue;
    CARTNode *parent;
};

FILE *fp_train, *fp_test, *fp_output, *fp_model;
int originTime, rg[NUM_THREADS+1];
int NClasses, NAttrs, MAttrs, NLines, NTrees;
float *_x;
int *_label;
int *_oobClass, hitCount, oobCount;
float oobError;
bool *_oobFlag;
CARTNode *_RGTreeHead;

/*
 * initialize a random forest
 *
 * NC: number of classes
 * NA: number of all attributes
 * MA: number of attributes selected to generate an individual tree
 * NL: number of records used to train the RF model
 * NT: number of trees
 */
void RFInit(int NC, int NA, int MA, int NL, int NT) {
    NClasses = NC;
    NAttrs = NA;
    MAttrs = MA;
    NLines = NL;
    NTrees = NT;

    _x = (float *)malloc(NLines*NAttrs*sizeof(float));
    _label = (int *)malloc(NLines*sizeof(int));
    _oobClass = (int *)malloc(NLines*NClasses*sizeof(int));
    _oobFlag = (bool *)malloc(NLines*sizeof(char));
    _RGTreeHead = (CARTNode *)malloc(NTrees*sizeof(CARTNode *));

    memset(_oobClass, 0, NLines*NClasses*sizeof(int));
    memset(_oobFlag, 0, NLines*sizeof(char));
    memset(_RGTreeHead, 0, NTrees*sizeof(CARTNode *));
}

void RFUninit() {
    fclose(fp_train);
	fclose(fp_test);
	fclose(fp_output);
	fclose(fp_model);
}

/*
 * read records from train file into memory, specify the file to save RF model
 */
void fillData(char *trainData, char *modelData) {
	fp_train = fopen(trainData, "r");
	fp_model = fopen(modelData, "w");

    float (*x)[NAttrs] = (float (*)[NAttrs])_x;
    int *label = _label;
 
    int id;
    char line[6400];
    fgets(line, 6400, fp_train);
    for (int i=0; i<NLines; i++) {
        fscanf(fp_train, "%d", &id);
        for (int j=0; j<NAttrs; j++) {
            fscanf(fp_train, ",%f", &x[i][j]);
        }
        fscanf(fp_train, ",%d", &label[i]);
    }
}

/*
 * generate a series of different trees
 *
 * args: pointer to data which specifies trees ID
 */
void *train(void* args) {
    int sampleID[NLines], attrID[MAttrs];
    bool sampleFlag[NLines], attrFlag[NAttrs];
    int cl[NClasses], countLeft[NClasses+1], countRight[NClasses+1];

    float (*x)[NAttrs] = (float (*)[NAttrs])_x;
    int *label = _label;
    int (*oobClass)[NClasses] = (int (*)[NClasses])_oobClass;
    bool *oobFlag = (bool *)_oobFlag;
    CARTNode **RGTreeHead= (CARTNode **)_RGTreeHead;

    srand(time(NULL));
    int from = *((int*)args), to = *((int*)args+1);
    // generate a series of trees
    for (int i=from; i<to; i++) {
        printf("Beginning build tree: %d, %ds\n", i, time(NULL)-originTime);

        // randomly select samples from training data with replacement
        memset(sampleFlag, 0, sizeof(sampleFlag));
        for (int j=0; j<NLines; j++) {
            sampleID[j] = rand()%NLines;
            sampleFlag[sampleID[j]] = true;
        }

        // randomly select a portion of attributes
        memset(attrFlag, 0, sizeof(attrFlag));
        for (int j=0; j<MAttrs; j++) {
            attrID[j] = rand()%NAttrs;
            if (attrFlag[attrID[j]])
                j--;
            else
                attrFlag[attrID[j]] = true;
        }

        printf("\tAfter bootstrap: %d, %ds\n", i, time(NULL)-originTime);

        deque<Interval> dq;
        Interval iv;
        dq.push_back(Interval(0, 0, NLines-1, -1.0, NULL));
        while (!dq.empty()) {
            iv = dq.front();
            dq.pop_front();

            if (iv.from > iv.to) { // empty leaf node
                CARTNode *leafNode = new CARTNode(-1, iv.parentValue, NULL, NULL);
                if (iv.parent->left == NULL) {
                    iv.parent->left = leafNode;
                } else {
                    iv.parent->right = leafNode;
                }
                continue;
            } else if (iv.to-iv.from<MIN_NODE_SIZE || iv.level>=TREE_DEPTH-1) { // small leaf node or depth is too deep
                int target = -1, maxTimes = -1;

                memset(cl, 0, sizeof(cl));
                for (int k=iv.from; k<=iv.to; k++) {
                    cl[label[sampleID[k]]-1]++;
                }
                for (int k=0; k<NClasses; k++) {
                    if (cl[k] > maxTimes) {
                        target = k;
                        maxTimes = cl[k];
                    }
                }

                CARTNode *leafNode = new CARTNode(-1, target, NULL, NULL);
                if (iv.parent->left == NULL) {
                    iv.parent->left = leafNode;
                } else {
                    iv.parent->right = leafNode;
                }
                continue;
            }

            int range = iv.to-iv.from+1;
            int NSplitValue = range<NUM_SPLIT_ENUM ? range : NUM_SPLIT_ENUM+range/300;
            int optAttr;
            float optSplitValue, minGini = 1.0e+100;
            // select the optimum attribute and its split value based on gini
            for (int k=0; k<MAttrs; k++) {
                for (int u=0; u<NSplitValue; u++) {
                    int idx = iv.from+rand()%range; // sampling due to too many values
                    float gini = 0.0, giniLeft = 1.0, giniRight = 1.0, pk;

                    memset(countLeft, 0, sizeof(countLeft));
                    memset(countRight, 0, sizeof(countRight));

                    for (int v=iv.from; v<=iv.to; v++) {
                        if (x[sampleID[v]][attrID[k]] <= x[sampleID[idx]][attrID[k]]) {
                            countLeft[NClasses]++;
                            countLeft[label[sampleID[v]]-1]++;
                        } else {
                            countRight[NClasses]++;
                            countRight[label[sampleID[v]]-1]++;
                        }
                    }
                    if (countLeft[NClasses]) {
                        for (int v=0; v<NClasses; v++) {
                            pk = (float)countLeft[v]/countLeft[NClasses];
                            giniLeft -= pk*pk;
                        }
                    }
                    if (countRight[NClasses]) {
                        for (int v=0; v<NClasses; v++) {
                            pk = (float)countRight[v]/countRight[NClasses];
                            giniRight -= pk*pk;
                        }
                    }
                    float pL = (float)countLeft[NClasses]/(countLeft[NClasses]+countRight[NClasses]);
                    gini = pL*giniLeft+(1-pL)*giniRight;

                    if (gini < minGini) {
                        optAttr = attrID[k];
                        optSplitValue = x[sampleID[idx]][attrID[k]];
                        minGini = gini;
                    }
                }
            }

            // create a new split node through the optimum attribute
            CARTNode *curNode = new CARTNode(optAttr, optSplitValue, NULL, NULL);
            if (RGTreeHead[i] == NULL) {
                RGTreeHead[i] = curNode;
            } else if (iv.parent->left == NULL) {
                iv.parent->left = curNode;
            } else {
                iv.parent->right = curNode;
            }

            // 1. take consideration of an empty child node, which needs its parent's predictive value
            // 2. split data in current node into left and right node
            int mid = iv.from, tmp;
            int target = -1, maxTimes = -1;
            memset(cl, 0, sizeof(cl));
            for (int v=iv.from; v<=iv.to; v++) {
                cl[label[sampleID[v]]-1]++;
                if (x[sampleID[v]][optAttr] <= optSplitValue) {
                    tmp = sampleID[v];
                    sampleID[v] = sampleID[mid];
                    sampleID[mid] = tmp;
                    mid++;
                }
            }
            for (int k=0; k<NClasses; k++) {
                if (cl[k] > maxTimes) {
                    target = k;
                    maxTimes = cl[k];
                }
            }

            dq.push_back(Interval(iv.level+1, iv.from, mid-1, target, curNode));
            dq.push_back(Interval(iv.level+1, mid, iv.to, target, curNode));
        }

        printf("\tFinished building tree: %d, %ds\n", i, time(NULL)-originTime);

        // help to caculate oob error
        for (int j=0; j<NLines; j++) {
            if (!sampleFlag[j]) {
                oobFlag[j] = true;

                CARTNode *iter = RGTreeHead[i];
                while (iter->attr != -1) {
                    if (x[j][iter->attr] <= iter->value)
                        iter = iter->left;
                    else
                        iter = iter->right;
                }

                oobClass[j][(int)(iter->value+0.1)]++;
            }
        }
    }

    printf("All trees (%d-%d) are built: %ds\n", from, to, time(NULL)-originTime);
}

/*
 * caculate the oob(out of bag) error rate
 */
void calOobError() {
    int *label = _label;
    int (*oobClass)[NClasses] = (int (*)[NClasses])_oobClass;
    bool *oobFlag = (bool *)_oobFlag;

    for (int j=0; j<NLines; j++) {
        if (oobFlag[j]) {
            oobCount++;

            int cls = -1, maxTimes = 0;
            for (int k=0; k<NClasses; k++) {
                if (oobClass[j][k] > maxTimes) {
                    cls = k;
                    maxTimes = oobClass[j][k];
                }
            }
            if (cls == label[j]-1)
                hitCount++;
        }
    }
    oobError = 1.0-(float)hitCount/oobCount;

    printf("Oob error caculated finished: %ds\n\n", time(NULL)-originTime);
    printf("oob error: %.2f%%\n", oobError*100);
}

/*
 * predict the class of a given record
 */
int predict(float *tx) {
    int classCount[NClasses];
    CARTNode **RGTreeHead = (CARTNode **)_RGTreeHead;

    memset(classCount, 0, sizeof(classCount));
    // vote of each classification tree
    for (int i=0; i<NTrees; i++) {
        CARTNode *iter = RGTreeHead[i];
        while (iter->attr != -1) {
            if (tx[iter->attr] <= iter->value)
                iter = iter->left;
            else
                iter = iter->right;
        }

        classCount[(int)(iter->value+0.1)]++;
    }

    int finalCls = -1, maxCount = 0;
    // make the class with most vote as target
    for (int i=0; i<NClasses; i++) {
        if (classCount[i] > maxCount) {
            finalCls = i;
            maxCount = classCount[i];
        }
    }

    return finalCls;
}

/*
 * read records from test file, make predictions and write the results into output file
 *
 * LC: the number of records in test file
 */
void predictFile(char *testData, int LC, char *outputData) {
	fp_test = fopen(testData, "r");
	fp_output = fopen(outputData, "w");
    
    float tx[NAttrs];
    int label;
 
    int id;
    char line[6400];
    fgets(line, 6400, fp_test);
    fprintf(fp_output, "id,label\n");
    for (int i=0; i<LC; i++) {
        fscanf(fp_test, "%d", &id);
        for (int j=0; j<NAttrs; j++) {
            fscanf(fp_test, ",%f", &tx[j]);
        }
        label = predict(tx)+1;
        fprintf(fp_output, "%d,%d\n", id, label);
    }
}

int main()
{
    originTime = time(NULL);

    rg[0] = 0;
    for (int i=1; i<NUM_THREADS; i++) {
        rg[i] = rg[i-1]+NUM_TREES/NUM_THREADS;
        if (i <= NUM_TREES%NUM_THREADS)
            rg[i]++;
    }
    rg[NUM_THREADS] = NUM_TREES;

    RFInit(26, 617, M_ATTRS, 6238, NUM_TREES);
    fillData(TRAIN_DATA, MODEL_DATA);
    
    // create threads to generate different trees
    pthread_t tids[NUM_THREADS];
    for(int i=0; i<NUM_THREADS; i++) {
        int ret = pthread_create(&tids[i], NULL, train, (void*)&rg[i]);
        if(ret != 0) {
            printf("pthread_create error: error_code=%d\n", ret);
        }
    }
    for(int i=0; i<NUM_THREADS; i++) {
        pthread_join(tids[i], NULL);
    }

    calOobError();
    predictFile(TEST_DATA, 1559, OUTPUT_DATA);
    RFUninit();

	return 0;
}
