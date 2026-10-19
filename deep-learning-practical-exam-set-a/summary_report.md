# Summary Report: Deep Learning Practical Exam (Set A)

**Author:** Maitrak

## Tasks and datasets

I built five standalone mini-projects, one for each core Deep Learning topic.

1. **ANN:** predicting diabetes from the Pima Indians Diabetes dataset (768 patients, 8 medical features).
2. **OpenCV:** a preprocessing pipeline on five wildlife photos (deer, lion, leopard, fox and tiger). Each image was resized, converted to grayscale, blurred, edge-detected with Canny and boxed around its largest contour.
3. **CNN:** classifying handwritten digits from a subset of MNIST (20,000 training and 4,000 test images). The exam suggests CIFAR-10, but its download was blocked on my network, so I switched to MNIST, which has the same 10-class structure.
4. **RNN:** forecasting monthly airline passengers (144 months, 1949 to 1960) from the previous 12 months.
5. **Transformer:** sentiment classification of 25 movie-review sentences I wrote, using a pre-trained DistilBERT model without any training.

## New concept I learned

The idea that was newest to me was how to treat time-series data correctly. For the RNN I had to turn one long column of numbers into sliding windows of 12 months, reshape them into (samples, timesteps, features), and split them in time order without shuffling. Shuffling would leak future values into training and make the results look better than they really are. I also learned to fit the scaler on the training data only.

I also saw how powerful pre-trained Transformers are: a model I never trained classified all 20 clear sentences correctly.

## Results on each task

- **ANN:** test accuracy was 69.5%, with precision 0.57, recall 0.54 and F1 0.55. Validation accuracy reached about 80% during training, so the lower test score reflects the small test set of 154 patients and the noisy data.
- **OpenCV:** all five images were processed through every stage and the results are saved in the outputs folder, including a 5x5 comparison grid. Canny sometimes boxed background texture instead of the whole animal.
- **CNN:** test accuracy was 97.5% with a macro F1 of 0.97. Training and validation curves stayed close, so there was little overfitting. Digit 9 had the lowest recall at 0.93.
- **RNN:** test RMSE was 43.99 passengers. The test period has higher passenger counts than the training period, which a simple RNN struggles to extrapolate.
- **Transformer:** 20 out of 20 clear sentences were correct. On the five ambiguous sentences the model still answered with 96 to 99% confidence, because it has no neutral label.

## What I would build next

With more time and compute, I would improve each model in turn. For the ANN I would add Dropout, tune the layer sizes and use class weights to raise recall. For the RNN I would try an LSTM and remove the trend from the series. I would move the CNN to full CIFAR-10 with data augmentation and then serve it through a small API. Finally, I would fine-tune the Transformer on Indian-English reviews and add a neutral class, so it can express uncertainty on ambiguous text.