# PeaceMaker

PeaceMaker is an experimental Federated Learning setup to demonstrate a novel flow with incentive capabilities for cross-silo Federated Learning. To learn more about the project in depth, refer to the [wiki](https://github.com/akassharjun/peacemaker/wiki/Project-Brief) page.

## Components

### User Interface

The user interface presents the data about previous training rounds, the model-data valuation, the payout value & current statistics of the global model. 
The Federated Learning process is started by initiating the training session via the UI by a system administrator.

### API Gateway

The API gateway is as a service that serves information presented on the UI and acts as the middleman between the UI and the Federated Learning server.

### Federated Learning server

An experimental and basic server setup to mimic Federated Learning between multiple organisations with the use of PySyft.



## External Links
- [Thesis](https://drive.google.com/file/d/1bb05WTNSgj42xEbE7MH6LKW1mtOMs9rl/view?usp=sharing)

