#include <iostream>
#include <string>
#include <cstdlib>

#include "config.h"
#include "manager.h"

using namespace std;

void usage();

int main(int argc, char* argv[]) {
    if (argc < 3) {
        usage();
        return 0;
    }
    int mode = atoi(argv[2]);
    if (mode == 0) {
        if (argc != 7) {
            cout << "./gibbs -m 0 -s snapshot -c config" << endl;
            exit(1);
        }
        string fn_snapshot = argv[4];
        string fn_config = argv[6];
        Config config;
        if (!config.Load(fn_config)) {
            cout << "Cannot load configuration file." 
                << " Check " << fn_config << endl;
        }
        else {
            cout << "Configuration file loaded successfully." << endl;
        }
        config.print();
        Manager manager(&config);
        if (fn_snapshot == "") {
            cout << "No file model is specified. Need a previous snapshot file" << endl;
        }
        else {
            manager.InitializeModel(fn_snapshot);
            cout << "Model has shown successfully" << endl;
        }
    }
    else if (mode == 1) {
        if (argc != 19 && argc != 21) {
            usage();
            return -1;
        }
        string fn_list = argv[4];
        string fn_config = argv[6];
        int n_iter = atoi(argv[8]);
        string fn_gaussian = argv[10];
        string basedir = argv[12];
        int batch_size = atoi(argv[14]);
        string fn_gseed = argv[16];
        string fn_sil = ""; 
        string fn_snapshot = "";

        if (argc == 19) {
            fn_sil = argv[18];
        }
        if (argc == 21) {
            fn_sil = argv[18];
            fn_snapshot = argv[20];
        }

        Config config;
        if (!config.Load(fn_config, fn_gaussian)) {
            cout << "Cannot load configuration file." 
                << " Check " << fn_config << endl;
        }
        else {
            cout << "Configuration file loaded successfully." << endl;
        }
        config.print();
        Manager manager(&config);
        if (mode == 0) {
        }
        else if (mode == 1) {
            if (fn_sil != "") {
                if (!manager.LoadSilenceModel(fn_sil)) {
                    cout << "Cannot load silence model " 
                        << "Check " << fn_sil << endl;
                }
                else {
                    config.set_silence(true);
                }
            }
            else {
                cout << "Training model without using silence model" << endl;
            }
            if (!config.LoadSeedingMixtures(fn_gseed)) {
                cout << "Cannot load Gaussian Seeding models" 
                    << " Check " << fn_gseed << endl;
                exit(1);
            }
            if (fn_snapshot == "") {
                manager.InitializeModel();
            }
            else {
                manager.InitializeModel(fn_snapshot);
            }

            if (!manager.LoadData(fn_list)) {
            cout << "Cannot load bounds" 
                << " Check " << fn_list << endl; 
            }
            else {
                cout << "Data loaded successfully." << endl;
            }
            manager.Inference(batch_size, n_iter, basedir);
        }
        else {
            cout << "Undefined mode: [0: read model; 1: training]" << endl;
        }
        cout << "Returning" << endl;
    }
    return 0;
}

void usage() {
    cout << "gibbs -m [0: read model; 1: training] -l data_list -c configuration -n num_iteration " 
        << "-g gaussian_prior -b basedir -z batch_size -sd gaussian_seeds -s silence_model -snapshot snapshot_file" << endl;
}
