Early Stopping
this is technique used to avoid overfitting - <img width="372" height="248" alt="image" src="https://github.com/user-attachments/assets/1efbdef4-c637-472e-8fd9-44f1af48125b" />

early stoppping technique stop training of model when it overfits (not run further epoches)
it use callback function to stop traning - 
callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False
)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3500, callbacks=callback)
