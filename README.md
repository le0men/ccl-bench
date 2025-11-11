# CCL-Bench

![CCL-Bench logo](./assets/logo.png)


We are going to pursue a class-wide benchmarking endeavor! As ML models get larger and larger, various distributed implementations and techniques have been proposed in industry and academia. However, a distributed implementation on hardware A using communication library X may behave drastically differently from the implementation on hardware B with communication library Y. Different implementations may lead to different scaling challenges. To better understand the nuances and to gain better insight on the challenges involved, we are planning to construct a benchmark, evaluating a wide variety of models on various frameworks and hardware types. The end goal is building a benchmark, CCL-Bench, which the community could benefit from.


## Initialization
```
conda create --name ccl-bench python=3.10.12
conda activate ccl-bench
pip install -r requirements.txt
```

## Process FLow
1. Select a workload from `./workloads`
2. Select suitable dataset, batch size, sequence length, etc., and specify the infrastructure (execution environment) listed in `workload_card_template.yaml`.
3. Determine suitbale exeuction plan (parallelization strategy, communication backend selection, etc.) for the workload and framework selected, and specify those choices in `workload_card_template.yaml`.
4. Profile and collect traces by following the guidelines in `trace_gen`
5. Store traces, copy the model card template, fill in the fields, store it under `trace_collection/<workload_name>`, and upload the traces to Google drive.
6. Define metrics (may happen before step 3)
7. Develop tools, and store it under `tools`
8. Calculate metrics
9. Upload metrics

## Layout
```
├── README.md
├── requirements.txt
├── workload_card_template.yaml # metadata template, should be located in trace_collection/<workload> folder
├── scripts  # scripts to execute tools for different metrics
├── tools   # main.py, and various plug-ins for different metrics
└── trace_collection # place to store temparary traces locally, which are downloaded from Google Drive.
```