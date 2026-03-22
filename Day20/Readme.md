Early Stopping
this is technique used to avoid overfitting - <img width="372" height="248" alt="image" src="https://github.com/user-attachments/assets/1efbdef4-c637-472e-8fd9-44f1af48125b" />

early stoppping technique stop training of model when it overfits (not run further epoches)
it use callback function to stop traning - 
callback = EarlyStopping(
    monitor="val_loss", Quantity to be monitored. Defaults to "val_loss"
    
    min_delta=0.00001, Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no                                improvement. Defaults to 0
    patience=20, when no changes in critera it patientely check next 20 epoches is improvement or not
    verbose=1, Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages when the callback takes an action. Defaults to 0.
    mode="auto", One of {"auto", "min", "max"}. In min mode, training will stop when the quantity monitored has stopped decreasing; in "max" mode it will stop when the quantity monitored has stopped increasing; in "auto" mode, the direction is automatically inferred from the name of the monitored quantity. Defaults to "auto".
    baseline=None, Baseline value for the monitored quantity. If not None, training will stop if the model doesn't show improvement over the baseline. Defaults to None.
    restore_best_weights=False Number of epochs to wait before starting to monitor improvement. This allows for a warm-up period in which no improvement is expected and thus training will not be stopped. Defaults to 0.
)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3500, callbacks=callback)
