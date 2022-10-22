# PeaceMaker

## Background

### FEDERATED LEARNING
Technology that allows machine learning models to be trained on edge devices, preserving data privacy and allowing us to apply machine learning to previously data sensitive usecases.

### DIFFERENT SETTINGS IN FEDERATED LEARNING
Federated Learning provides two different settings in which it can be applied; the cross-device setting & the cross-silo setting. 

#### Cross-Device

The cross-device setting in FL typically revolves around a service provider and its consumers. The clients in this case are usually mobile phones or Internet of Things devices. These clients are responsible for collaboratively training a model for the service provider. The best example of cross-device is Google's Keyboard. refer this [link](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html).

#### Cross-Silo

The main idea behind cross-silo is to unlock the potential of data stored away in huge repositories under large organisations. For example, we could enable the collaboration of organisations like Hospitals or Banks to build a global machine learning model.

### EFFECTIVENESS OF FEDERATED LEARNING
The effectiveness of a Federated Learning system relies on various aspects, one of which is the most important; client participation. 

## Problem

### SCENARIO
An organisation A would like to leverage Machine Learning to solve a certain problem for their users & boost their business, but it relies on data that is scarce, & private. 

### SOLUTION
Leverage Cross-Silo Federated Learning which solves the data privacy issue and collaborating with many organisations B, C & D should help dealing with the scarcity.

### PROBLEM IN THE SOLUTION!
The collaboration might benefit organisation A’s business but do the other organisations really benefit from this overall? 

## Literature Support

- “Clients might worry that contributing their data to training federated learning models will benefit their competitors, who do not contribute as much but receive the same final model nonetheless” (Kairouz et al., 2021)
- “When the parties are competing with one another, they maybe unwilling to participate in the federated learning since the other competitors can also beneﬁt from their own contributions.” (Zhan et al., 2021)
- “Researchers should put more emphasis on cross-silo FL. The decision behavior of large company/organization is distinct from that of end users and mobile devices, which further requires a totally new incentive method for cross-silo FL. .” (Zeng et al., 2021)

## Research Questions 

### RESEARCH QUESTION #1
How do we drive organisations to collaborate together amidst the conflict of interests? 

#### AIM 1
Design a new flow for cross-silo Federated Learning with incentive capabilities.

### RESEARCH QUESTION #2
How can we measure the dataset quality of an organization?

#### AIM 2
Identify means to quantify dataset quality.

### RESEARCH QUESTION #3
How would this incentive be sourced?

#### AIM 3
Identify means to source the incentive.

## External Links
- [Thesis](https://drive.google.com/file/d/1bb05WTNSgj42xEbE7MH6LKW1mtOMs9rl/view?usp=sharing)

## References

Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., Bonawitz, K., Charles, Z., Cormode, G., Cummings, R., D’Oliveira, R. G. L., Eichner, H., Rouayheb, S. E., Evans, D., Gardner, J., Garrett, Z., Gascón, A., Ghazi, B., Gibbons, P. B., Gruteser, M., Harchaoui, Z., He, C., He, L., Huo, Z., Hutchinson, B., Hsu, J., Jaggi, M., Javidi, T., Joshi, G., Khodak, M., Konečný, J., Korolova, A., Koushanfar, F., Koyejo, S., Lepoint, T., Liu, Y., Mittal, P., Mohri, M., Nock, R., Özgür, A., Pagh, R., Raykova, M., Qi, H., Ramage, D., Raskar, R., Song, D., Song, W., Stich, S. U., Sun, Z., Suresh, A. T., Tramèr, F., Vepakomma, P., Wang, J., Xiong, L., Xu, Z., Yang, Q., Yu, F. X., Yu, H. and Zhao, S. (2021), ‘Advances and open problems in federated learning’.

Zhan, Y., Zhang, J., Hong, Z., Wu, L., Li, P. and Guo, S. (2021), ‘A survey of incentive mechanism design for federated learning’, IEEE Transactions on Emerging Topics in Computing pp. 1–1.

Zeng, R., Zeng, C., Wang, X., Li, B. and Chu, X. (2021), ‘A comprehensive survey of incentive mechanism for federated learning’, ArXiv abs/2106.15406.
