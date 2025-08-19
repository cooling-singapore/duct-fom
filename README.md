# DUCT Federation of Models

Cooling Singapore's Digital Urban Climate Twin (DUCT) makes use of a variety of models needed to simulate the urban climate. These models are utilised by the [DUCT Explorer](https://github.com/cooling-singapore/duct-explorer) to conduct various analyses. In order for these analyses to utilise the models, they need to be onboarded to be compatible with the [Simulation-as-a-Service Middleware](https://github.com/cooling-singapore/saas-middleware) developed by Cooling Singapore for the purpose of operationalising models. For this purpose, adapters are used that 'wrap' the underlying modelling software. This repository contains the adapters used for the DUCT models.

Modelling software can be highly complex and involve several distinct steps that are required in order to carry out a simulation. Adapters are used to automate and package multiple of these steps together. For practical reasons it often makes sense to package all pre-processing steps into one adapter while having the actual simulation execution in a separate adapter. This can have advantages to deploy the adapters in different environments according to the technical requirements of the steps involved. As a result, there are typically multiple adapters for each modelling software.

The following third party model systems have been onboarded through corresponding Simulation-as-a-Service adapters:
- [WRF (Weather Research and Forecasting) for mesoscale urban climate modelling](ucm-wrf/README.md).
- [Palm for microscale urban climate modelling](ucm-palm/README.md).
- [CEA (City Energy Analyst) for building energy system modelling](bem-cea/README.md).
- [OpenFOAM-based Scout model by A*STAR for microscale urban climate modelling](ucm-scout/README.md).

Cooling Singapore researchers developed custom models that have also been onboarded using corresponding Simulation-as-a-Service adapters:
- [MVA (Multivariate Analysis) for urban ventilarion and wind corridor modelling](ucm-mva/README.md).
