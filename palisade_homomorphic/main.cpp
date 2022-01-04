#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>

#include "palisade.h"
#include "ciphertext-ser.h"
#include "scheme/ckks/ckks-ser.h"
#include "cryptocontext-ser.h"
#include "pubkeylp-ser.h"


using namespace std;
using namespace lbcrypto;
const string DATAFOLDER = "demoData";


double normGen(vector<double> x) {
    double magnitude = 0;
    for (auto& n : x) {
        magnitude += pow(n, 2);
    }
    magnitude = 1 / sqrt(magnitude);
    return magnitude;
}

int main(int argc, char *argv[]) {
    double diff, start, finish, time1, time2;

    cout << "Sample mean vector generation using CKKS" << endl;

    uint32_t multDepth = 3; // how many multiplications can we perform on a ciphertext.
    uint32_t scaleFactorBits = 50; //precision of computations, CKKS erases 12-25 LSB, we may loose up to 1 bit after each operation

    uint32_t batchSize = 128; // size of the vector
    int batch = 128;

    SecurityLevel securityLevel = HEStd_128_classic; // HEStd_192_classic, HEStd_256_classic

    start = currentDateTime();
    // Step 1: Context Generation
    CryptoContext<DCRTPoly> cryptoContext =
        CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
            multDepth, scaleFactorBits, batchSize, securityLevel);

    finish = currentDateTime();
    diff = finish - start;
    cout << "\nParam generation time: " << "\t" << diff << " ms" << endl;

    cout << "CKKS scheme is using ring dimension " << cryptoContext->GetRingDimension()
        << endl
        << endl;

    cryptoContext->Enable(ENCRYPTION);
    cryptoContext->Enable(SHE);
    cryptoContext->Enable(LEVELEDSHE);

    cout << "p = "
        << cryptoContext->GetCryptoParameters()->GetPlaintextModulus()
        << endl;
    cout << "n = "
        << cryptoContext->GetCryptoParameters()
                        ->GetElementParams()
                        ->GetCyclotomicOrder() /
                   2
        << endl;
    cout << "log2 q = "
        << log2(cryptoContext->GetCryptoParameters()
                             ->GetElementParams()
                             ->GetModulus()
                             .ConvertToDouble())
        << endl;
    cout << "r = " << cryptoContext->GetCryptoParameters()->GetRelinWindow()
        << endl;

    // Step 2: Public/Secret Key Generation
    cout << "\nRunning key generation..."
            << endl;
    start = currentDateTime();
    auto keyPairUser = cryptoContext->KeyGen();

    finish = currentDateTime();
    diff = finish - start;
    cout << "Querier --> Key generation time: " << "\t" << diff << " ms" << endl;


    if (!keyPairUser.good()) {
        cout << "Key generation failed!" << endl;
        exit(1);
    }

    // Step 3: Evaluation Keys Generation
    start = currentDateTime();
    cryptoContext->EvalMultKeyGen(keyPairUser.secretKey); // Multiplication key
    cryptoContext->EvalSumKeyGen(keyPairUser.secretKey); // Sum key
    cryptoContext->EvalAtIndexKeyGen(keyPairUser.secretKey, {-1}); // Rotation key, careful of the number of rotation keys.
    finish = currentDateTime();
    diff = finish - start;
    cout << "Querier --> EvalKeys (Mult, Sum, Rotation) generation time: " << "\t" << diff << " ms" << endl;

    // Sample Vectors
    vector<double> x1 = {0.09941268712282181, -0.010987147688865662, 0.04565052688121796, 0.14781123399734497, 0.08896120637655258, 0.2463078498840332, 0.022987842559814453, -0.2725674510002136, 0.2141762375831604, 0.17619910836219788, 0.014700625091791153, -0.10778379440307617, 0.27074500918388367, 0.048202771693468094, -0.17793191969394684, 0.016743503510951996, 0.34787148237228394, -0.24206185340881348, -0.11656205356121063, -0.3236809968948364, 0.2827132046222687, -0.021762117743492126, 0.2886936068534851, -0.22026003897190094, 0.17998678982257843, -0.05641006678342819, -0.27283236384391785, -0.03135531395673752, 0.1838061809539795, 0.07630791515111923, -0.037367310374975204, -0.036986641585826874, -0.11924898624420166, -0.2886181175708771, 0.01566002517938614, -0.22661173343658447, 0.04744769260287285, 0.19079545140266418, 0.07220453023910522, 0.12298743426799774, 0.1951199471950531, 0.024730591103434563, 0.16199669241905212, -0.1956695020198822, 0.0007139829103834927, -0.15586651861667633, 0.2657493054866791, 0.08966108411550522, -0.06911837309598923, -0.2938316762447357, -0.017146648839116096, 0.22416247427463531, -0.021329572424292564, -0.02085276134312153, -0.19025367498397827, 0.01792571134865284, 0.00884596910327673, 0.19178172945976257, -0.10258624702692032, 0.009083596058189869, -0.01356898806989193, 0.10244820266962051, -0.015654059126973152, -0.07917102426290512, -0.009235094301402569, 0.18583820760250092, 0.11614726483821869, 0.12711890041828156, 0.09366777539253235, 0.21679732203483582, -0.18584044277668, -0.009261886589229107, 0.21796463429927826, -0.0966799408197403, 0.2520400881767273, 0.049208514392375946, -0.0072151911444962025, 0.09750401228666306, -0.06607112288475037, 0.0934421643614769, -0.07251360267400742, -0.05009816959500313, -0.03611046448349953, -0.07871882617473602, 0.11084847152233124, 0.1405927538871765, -0.25177985429763794, 0.3024395704269409, 0.155969500541687, 0.03204911947250366, -0.04414093866944313, 0.27279773354530334, -0.2921050786972046, 0.02282869815826416, 0.245872363448143, -0.047338373959064484, 0.22282591462135315, -0.014946673065423965, -0.149352565407753, -0.049864329397678375, 0.09366148710250854, -0.0816873237490654, 0.08927911520004272, -0.1098150834441185, 0.19080255925655365, 0.09589938074350357, 0.2056376338005066, 0.045242790132761, -0.09837073087692261, 0.1624271124601364, 0.039139773696660995, 0.1663196086883545, -0.18386560678482056, -0.004758839961141348, -0.11183617264032364, -0.1896277666091919, 0.02721957303583622, 0.09764956682920456, 0.0899343192577362, -0.14101192355155945, -0.13578912615776062, -0.14159467816352844, -0.2803057134151459, -0.043143730610609055, -0.24659022688865662, 0.10391002148389816, 0.13710498809814453, 0.07463202625513077};
    double magnitude1 = 0;
    for (auto& n : x1) {
        magnitude1 += pow(n, 2);
    }
    magnitude1 = 1 / sqrt(magnitude1);
    vector<double> norm1 = {magnitude1};

    std::ofstream out(DATAFOLDER + "/plaintext.txt");
    out<<x1;
    out.close();
    std::ofstream out1(DATAFOLDER + "/plaintextnorm.txt");
    out1<<norm1;
    out1.close();


    vector<double> x2 = {-0.213237464427948, -0.25666213035583496, 0.22461311519145966, 0.10549312829971313, 0.23016920685768127, -0.10120043903589249, -0.20922426879405975, 0.21426433324813843, 0.1195056363940239, 0.1252356916666031, 0.10129837691783905, 0.026435354724526405, 0.04056096076965332, 0.03353095054626465, -0.07831332087516785, 0.12838216125965118, 0.041042350232601166, 0.20842652022838593, -0.11516059935092926, -0.113819420337677, 0.1747535914182663, 0.04167409613728523, 0.010235057212412357, -0.08525613695383072, 0.11690354347229004, -0.18365684151649475, -0.17219777405261993, 0.14702750742435455, 0.24242931604385376, -0.12974750995635986, 0.05997868627309799, -0.1907787322998047, -0.1748953014612198, -0.19528882205486298, 0.0637163296341896, -0.06280630826950073, -0.10702040046453476, -0.0641327053308487, 0.034732017666101456, 0.11631903052330017, -0.1788206249475479, 0.16274431347846985, -0.0031971002463251352, 0.11064422875642776, 0.23088695108890533, -0.13279670476913452, 0.15031836926937103, 0.2634841203689575, 0.06772047281265259, 0.1759442538022995, -0.05132720246911049, 0.06764626502990723, -0.2186441272497177, -0.018719149753451347, -0.15887857973575592, -0.16345708072185516, -0.0712525025010109, -0.1430022269487381, -0.277482807636261, -0.13133004307746887, 0.12100540846586227, 0.2698369324207306, -0.10152328759431839, 0.022002940997481346, 0.10260702669620514, -0.12177450209856033, -0.051048628985881805, 0.29639679193496704, 0.17800892889499664, 0.04497561603784561, -0.17740635573863983, -0.2471763640642166, 0.047849252820014954, -0.22719141840934753, 0.09690003097057343, 0.08439972996711731, -0.2529991567134857, 0.01814207434654236, 0.11140740662813187, 0.20936797559261322, 0.0808221772313118, 0.02815866470336914, 0.1994326114654541, 0.14691226184368134, 0.08579555153846741, -0.045629799365997314, -0.06105918809771538, 0.19176335632801056, 0.2539813220500946, 0.27265340089797974, -0.24667240679264069, 0.2518443465232849, 0.15828752517700195, -0.13988877832889557, 0.10762115567922592, -0.23734746873378754, -0.23966805636882782, 0.13017638027668, -0.06178498640656471, 0.16880932450294495, 0.07098156958818436, -0.25553610920906067, 0.2092256397008896, -0.22759495675563812, 0.21280257403850555, -0.05004773661494255, 0.10478726029396057, 0.17781586945056915, -0.09905519336462021, 0.07290418446063995, 0.07047192007303238, 0.17087256908416748, -0.2454000562429428, -0.26863864064216614, -0.20134960114955902, -0.2528849244117737, -0.009618852287530899, -0.06784247606992722, 0.17245621979236603, -0.05942155793309212, 0.21978048980236053, -0.0955294743180275, -0.15977327525615692, -0.10219746828079224, -0.0041093723848462105, 0.17511814832687378, -0.2303849160671234, -0.2143431007862091};
    double magnitude2 = 0;
    for (auto& n : x2) {
        magnitude2 += pow(n, 2);
    }
    magnitude2 = 1 / sqrt(magnitude2);
    vector<double> norm2 = {magnitude2};

    vector<double> x3 = {-0.020945748314261436, 0.041765958070755005, 0.29016226530075073, -0.0381251685321331, 0.01412679348140955, -0.228329136967659, 0.04485686123371124, 0.25094589591026306, 0.16668809950351715, 0.2371417135000229, -0.19131268560886383, -0.0012573241256177425, -0.1403028666973114, -0.2193164825439453, -0.02012624405324459, 0.02654559351503849, -0.0581137016415596, 0.17755283415317535, 0.24775756895542145, 0.22660307586193085, -0.04615037888288498, -0.28539082407951355, -0.19501802325248718, 0.18371203541755676, -0.09431600570678711, -0.11117356270551682, 0.2890184223651886, -0.275741308927536, -0.22754241526126862, 0.1493486613035202, 0.15884028375148773, -0.039749857038259506, -0.24383202195167542, -0.23774513602256775, -0.13593147695064545, 0.06039753556251526, -0.186395525932312, -0.09912198036909103, 0.26868361234664917, 0.08581169694662094, -0.22720967233181, -0.12996484339237213, -0.058460623025894165, 0.05794583261013031, 0.16372931003570557, 0.1129712164402008, 0.08324600011110306, -0.17693734169006348, 0.21418771147727966, 0.24840524792671204, -0.15703611075878143, 0.13313420116901398, 0.006313917227089405, -0.133926123380661, 0.20528611540794373, 0.10307737439870834, -0.10988156497478485, 0.19121387600898743, 0.01939619891345501, -0.19092029333114624, 0.24440380930900574, 0.11649129539728165, -0.05707718804478645, -0.06231211870908737, 0.11410107463598251, -0.11366082727909088, -0.14120365679264069, -0.032398201525211334, 0.16007937490940094, -0.1931292563676834, -0.238491028547287, 0.001571540953591466, -0.04894651100039482, -0.1416558027267456, -0.15715141594409943, -0.2844752073287964, -0.22170627117156982, 0.2520546317100525, -0.1377040147781372, -0.13795053958892822, 0.26029330492019653, -0.24325117468833923, 0.1444653570652008, -0.164285346865654, -0.08623579889535904, -0.07844209671020508, 0.28051528334617615, -0.26289135217666626, 0.05911168083548546, -0.1813141107559204, 0.11084318906068802, 0.0923057273030281, 0.04195699468255043, 0.20450782775878906, 0.09461461752653122, -0.23460514843463898, -0.2911302149295807, -0.12149093300104141, 0.06849560141563416, 0.04221588000655174, 0.08937345445156097, -0.05907990783452988, 0.0536508783698082, 0.15932242572307587, -0.2646236717700958, -0.11168301105499268, -0.08148221671581268, 0.18593133985996246, -0.23949600756168365, 0.09089091420173645, -0.022707048803567886, 0.07377950102090836, -0.19085389375686646, -0.13499633967876434, -0.0965961292386055, 0.1407751441001892, -0.17248326539993286, -0.0983479842543602, 0.13248814642429352, 0.04661297798156738, -0.05383146554231644, -0.10005352646112442, 0.05640729144215584, -0.10178683698177338, 0.14049337804317474, 0.18556946516036987, -0.13511592149734497, -0.129453644156456};
    double magnitude3 = 0;
    for (auto& n : x3) {
        magnitude3 += pow(n, 2);
    }
    magnitude3 = 1 / sqrt(magnitude3);
    vector<double> norm3 = {magnitude3};

    // Step 4: Plaintext Encoding
    start = currentDateTime();
    Plaintext ptxt1 = cryptoContext->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt1Norm = cryptoContext->MakeCKKSPackedPlaintext(norm1);
    finish = currentDateTime();
    diff = finish - start;
    cout << "Querier --> Plaintext Encoding time: " << "\t" << diff << " ms" << endl;
    Plaintext ptxt2 = cryptoContext->MakeCKKSPackedPlaintext(x2);
    Plaintext ptxt2Norm = cryptoContext->MakeCKKSPackedPlaintext(norm2);
    Plaintext ptxt3 = cryptoContext->MakeCKKSPackedPlaintext(x3);
    Plaintext ptxt3Norm = cryptoContext->MakeCKKSPackedPlaintext(norm3);

    // It will be used from cryptoEvaluator for transforming cosine similarity into [0, 2]
    vector<double> mask = {1};
    Plaintext maskPtxt = cryptoContext->MakeCKKSPackedPlaintext(mask);

    /*
    cout << "Input x1: " << ptxt1 << endl;
    cout << "Inversed norm x1" << ptxt1Norm << endl;
    cout << "Input x2: " << ptxt2 << endl;
    cout << "Inversed norm x2" << ptxt2Norm << endl;
    cout << "Input x3: " << ptxt3 << endl;
    cout << "Inversed norm x3" << ptxt3Norm << endl;
    */

    // Step 5: Encryption of encoded data
    start = currentDateTime();
    auto ciphertext1 = cryptoContext->Encrypt(keyPairUser.publicKey, ptxt1);
    auto ciphertext1Norm = cryptoContext->Encrypt(keyPairUser.publicKey, ptxt1Norm);
    finish = currentDateTime();
    diff = finish - start;
    cout << "Querier --> Encryption time: " << "\t" << diff << " ms" << endl;


    auto ciphertext2 = cryptoContext->Encrypt(keyPairUser.publicKey, ptxt2);
    auto ciphertext2Norm = cryptoContext->Encrypt(keyPairUser.publicKey, ptxt2Norm);

    auto ciphertext3 = cryptoContext->Encrypt(keyPairUser.publicKey, ptxt3);
    auto ciphertext3Norm = cryptoContext->Encrypt(keyPairUser.publicKey, ptxt3Norm);

    // Step 6: Calculate Cosine Similarity
    /* Assuming that cryptoEvaluator has already collected vectors from the querier and their friends, it performs:
            EvalInnerProduct() to get the numerator
            EvalMult() to get the 'denominator'
            EvalMult() between numerator and denominator to get the cosine similarity
            EvalAdd() with the masked encoded plaintext to transform cosine similarity in [0, 2]*/
    start = currentDateTime();
    // Dot Product
    auto innerProduct = cryptoContext->EvalInnerProduct(ciphertext1, ciphertext2, batchSize);
    // Multiply inversed norms
    auto norms = cryptoContext->EvalMult(ciphertext1Norm, ciphertext2Norm);
    // Get Cosine sim
    auto cosineSim = cryptoContext->EvalMult(innerProduct, norms);

    cosineSim = cryptoContext->EvalAdd(cosineSim, maskPtxt);

    finish = currentDateTime();
    diff = finish - start;
    cout << "CryptoEvaluator --> Homomomorphic Cosine Similarity for a pair time: " << "\t" << diff << " ms" << endl;

    auto innerProduct1 = cryptoContext->EvalInnerProduct(ciphertext1, ciphertext3, batchSize);
    // Multiply inversed norms
    auto norms1 = cryptoContext->EvalMult(ciphertext1Norm, ciphertext3Norm);
    // Get Cosine sim
    auto cosineSim1 = cryptoContext->EvalMult(innerProduct1, norms1);
    cosineSim1 = cryptoContext->EvalAdd(cosineSim1, maskPtxt);

    start = currentDateTime();
    auto weighted = cryptoContext->EvalMult(ciphertext1, 2); // multiply the querier's vector by 2.
    finish = currentDateTime();
    diff = finish - start;
    cout << "CryptoEvaluator --> Homomomorphic multiplication for the querier's vector (Scalar Mult): " << "\t" << diff << " ms" << endl;

    start = currentDateTime();

    Ciphertext<DCRTPoly> cRot = cryptoContext->EvalAtIndex(cosineSim, -1); // right rotation
    Ciphertext<DCRTPoly> cRes;
    for (int i=1;i<batch;i++) {
        if (i == 1) {
            cRes = cryptoContext->EvalAdd(cosineSim, cRot);
        }
        else {
            cRot = cryptoContext->EvalAtIndex(cRot, -1); // continue rotating
            cRes = cryptoContext->EvalAdd(cRes, cRot);
        }
    }



    auto cWeighted1= cryptoContext->EvalMult(ciphertext2, cRes);

    finish = currentDateTime();
    diff = finish - start;
    cout << "CryptoEvaluator --> Homomomorphic multiplication for a friend's vector (rotations and Mult): " << "\t" << diff << " ms" << endl;

    cRot = cryptoContext->EvalAtIndex(cosineSim1, -1); // right rotation
    for (int i=1;i<batch;i++) {
        if (i == 1) {
            cRes = cryptoContext->EvalAdd(cosineSim1, cRot);
        }
        else {
            cRot = cryptoContext->EvalAtIndex(cRot, -1); // continue rotating
            cRes = cryptoContext->EvalAdd(cRes, cRot);
        }
    }

    auto cAdd2 = cryptoContext->EvalMult(ciphertext3, cRes);

    vector<Ciphertext<DCRTPoly>> cList; // We pack all ciphertexts inside a vector to perform EvalAddMany. We can always use EvalAdd.
    cList.push_back(weighted);
    cList.push_back(cWeighted1);
    cList.push_back(cAdd2);
    start = currentDateTime();
    auto cAdd = cryptoContext->EvalAddMany(cList);

    auto wAdd = cryptoContext->EvalAdd(cosineSim, cosineSim1);
    wAdd = cryptoContext->EvalAdd(wAdd, 2.0);

    finish = currentDateTime();
    diff = finish - start;
    cout << "CryptoEvaluator --> Homomomorphic addition for weighted vectors and sum of weights: " << "\t" << diff << " ms" << endl;

    start = currentDateTime();
    Plaintext weightedSum;
    cryptoContext->Decrypt(keyPairUser.secretKey, cAdd, &weightedSum);
    diff = currentDateTime();
    time1 = diff - start;

    Plaintext weightsSum;
    start = currentDateTime();
    cryptoContext->Decrypt(keyPairUser.secretKey, wAdd, &weightsSum);
    finish = currentDateTime();
    time2 = finish - start;

    cout << "Querier --> Decryption (vectors + weights): " << "\t" << time1+time2 << " ms" << endl;

    weightedSum->SetLength(128); // change me if you want to observe the whole resulting vector
    weightsSum->SetLength(1);
    cout << setprecision(17) << endl; // set precision to 17 to compare with plaintext results, see an example in l. 430
    cout << "Resulting Decryption for sum of weighted vectors \n" << weightedSum;
    cout << "Resulting Decryption for sum of weights \n" << weightsSum;


    // Serialize cryptoContext, keys, ciphertexts to measure storage requirements.
    cout << "\n\nSerializing..." << endl;

    if (!Serial::SerializeToFile(DATAFOLDER + "/cryptoContext.txt", cryptoContext, SerType::BINARY)) {
        cerr << "Error serializing the cryptocontext" << endl;
        return 1;
    }
    if (!Serial::SerializeToFile(DATAFOLDER + "/pkUser.txt", keyPairUser.publicKey, SerType::BINARY)) {
            cerr << "Error serializing pk" << endl;
            return 1;
    }
    if (!Serial::SerializeToFile(DATAFOLDER + "/skUser.txt", keyPairUser.secretKey, SerType::BINARY)) {
            cerr << "Error serializing sk" << endl;
            return 1;
    }
    if (!Serial::SerializeToFile(DATAFOLDER + "/ct1.txt", ciphertext1, SerType::BINARY)) {
        cerr << "Error serializing ct1" << endl;
        return 1;
    }
    if (!Serial::SerializeToFile(DATAFOLDER + "/ct2.txt", ciphertext2, SerType::BINARY)) {
        cerr << "Error serializing ct1" << endl;
        return 1;
    }
    if (!Serial::SerializeToFile(DATAFOLDER + "/ctNorm1.txt", ciphertext1Norm, SerType::BINARY)) {
        cerr << "Error serializing ct1" << endl;
        return 1;
    }

    if (!Serial::SerializeToFile(DATAFOLDER + "/ctNorm2.txt", ciphertext2Norm, SerType::BINARY)) {
        cerr << "Error serializing ct1" << endl;
        return 1;
    }

    if (!Serial::SerializeToFile(DATAFOLDER + "/cosineSim.txt", cosineSim, SerType::BINARY)) {
        cerr << "Error serializing cosineSim" << endl;
        return 1;
    }

    if (!Serial::SerializeToFile(DATAFOLDER + "/cWeighted1.txt", cWeighted1, SerType::BINARY)) {
        cerr << "Error serializing cWeighted1" << endl;
        return 1;
    }

    if (!Serial::SerializeToFile(DATAFOLDER + "/cAdd.txt", cAdd, SerType::BINARY)) {
        cerr << "Error serializing cAdd" << endl;
        return 1;
    }

    if (!Serial::SerializeToFile(DATAFOLDER + "/wAdd.txt", cAdd, SerType::BINARY)) {
        cerr << "Error serializing ct1" << endl;
        return 1;
    }

    ofstream multKeyFile(DATAFOLDER + "/multKey.txt", ios::out | ios::binary);
    if (multKeyFile.is_open()) {
        if(!cryptoContext->SerializeEvalMultKey(multKeyFile, SerType::BINARY)) {
            cerr << "Error writing EvalMult keys" << endl;
            return 1;
        }
        multKeyFile.close();
    }
    else {
        cerr << "Error serializing EvalMult keys" << endl;
        return 1;
    }

    ofstream sumKeyFile(DATAFOLDER + "/sumKey.txt", ios::out | ios::binary);
    if (sumKeyFile.is_open()) {
        if(!cryptoContext->SerializeEvalSumKey(sumKeyFile, SerType::BINARY)) {
            cerr << "Error writing EvalSum key" << endl;
            return 1;
        }
        sumKeyFile.close();
    }
    else {
        cerr << "Error writing EvalSum key" << endl;
        return 1;
    }

    ofstream rotationKeyFile(DATAFOLDER + "/rotKey.txt", ios::out | ios::binary);
    if (rotationKeyFile.is_open()) {
        if(!cryptoContext->SerializeEvalAutomorphismKey(rotationKeyFile, SerType::BINARY)) {
            cerr << "Error writing rotation keys" << endl;
            return 1;
        }
    }
    else {
        cerr << "Error writing rotation keys" << endl;
        return 1;
    }


}