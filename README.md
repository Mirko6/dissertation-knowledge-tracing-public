# part-II-project


Repository overview
```bash
├── assistment                                   # Related to correct version of the ASSISTment 2009-10 dataset
├── data                                         # Folder with datasets and train results
│   ├── assistment                               # Contains the ASSISTment 2009-10 dataset in different forms
│   ├── bkt_params                               # Trained bkt parameters
│   ├── isaac                                    # Isaac data in various forms, not published
│   ├── mc_nemar                                 # Predictions used in McNemar's test
│   ├── piech                                    # Piech's split data in various forms
│   ├── data_analyser.ipynb                      # Analysing Isaac and ASSISTment datasets
│   └── ...
├── evaluations                                  # Evaluations of KT models that didn't fit to other folders
│   ├── bkt_running_against_pyBKT.ipynb          # Comparing AUC and accuracies of my implementation with pyBKT 
│   ├── mc_nemar.ipynb                           # Running the McNemar's test on AKT vs. SAINT and AKT vs. DKT+
│   └── ...
├── factor_analysis                              # PFA implementation attempts
├── isaac                                        # Code related to processing and evaluation on the Isaac data
│   ├── answer_chunk_processor.ipynb             # Processes original Isaac interaction data       
│   ├── bktf_per_skill.ipynb                     # Train and eval corrected version of BKT-F
│   ├── pyBKT_on_isaac.ipynb                     # Tain and eval pyBKT models (BKT and BKT-F)
│   ├── question_processor.ipynb                 # Extracts important question-specific information such as KCs
│   ├── read_isaac_concept_labelled_data.ipynb   # Creates a final version of the Isaac Dataset
│   └── ...
├── main_package                                 # Contains BKT and BKT-F implementations and utils
│   ├── bkt_pyKT_per_skill_forget_fix.py         # Corrected version of BKT-F on pyKT formatted data
│   ├── bkt_pyKT_per_skill.py                    # BKT on pyKT formatted data
│   ├── bkt_pyKT.py                              # BKT and BKT-F with global parameters (not KC specific)
│   ├── bkts.py                                  # Initial BKT and BKT-F implementations with global parameters
│   ├── utils.py                                 # Contains code that was repeatedly used
│   └── ...
├── other_scripts                                # Contains other scripts that weren't used in the end
├── piech                                        # Relates to Piech's split of the ASSISTment 2009-10 dataset
│   ├── bkt_standard.ipynb                       # Train and evaluate BKT 
│   ├── bktf_pykt_format.ipynb                   # Train and evaluate BKT-F
│   └── ...
├── pyBKT                                        # Code relating to pyBKT such as evaluation on Piech's split
├── pyKT_evals                                   # Google Colabs in which pyKT models are run
├── License                                      # MIT License
└── ...

```
