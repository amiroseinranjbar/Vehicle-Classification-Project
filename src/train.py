import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs=150,
    patience=20,
    criterion=None
):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    best_train_loss = 0.0
    best_train_accuracy = 0.0
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            train_total += labels.size(0)

            train_correct += (
                predicted == labels
            ).sum().item()

        train_loss = (
            running_loss
            / train_total
        )

        train_accuracy = (
            train_correct
            / train_total
        )

        model.eval()

        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_running_loss += (
                    loss.item()
                    * labels.size(0)
                )

                _, predicted = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        val_loss = (
            val_running_loss
            / total
        )

        val_accuracy = (
            correct
            / total
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(
            train_accuracy
        )

        val_accuracies.append(
            val_accuracy
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%, "
            f"Val Accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_train_loss = train_loss

            best_train_accuracy = (
                train_accuracy
            )

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print("Early stopping.")

            break

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accuracy": train_accuracies,
        "val_accuracy": val_accuracies
    }

    return (
        best_val_accuracy,
        best_train_loss,
        best_train_accuracy,
        best_model_state,
        history
    )


def train_model_weight_scheduler(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs=150,
    patience=20,
    criterion=None
):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    best_train_loss = 0.0
    best_train_accuracy = 0.0
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            train_total += labels.size(0)

            train_correct += (
                predicted == labels
            ).sum().item()

        train_loss = (
            running_loss
            / train_total
        )

        train_accuracy = (
            train_correct
            / train_total
        )

        model.eval()

        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_running_loss += (
                    loss.item()
                    * labels.size(0)
                )

                _, predicted = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        val_loss = (
            val_running_loss
            / total
        )

        val_accuracy = (
            correct
            / total
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(
            train_accuracy
        )

        val_accuracies.append(
            val_accuracy
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%, "
            f"Val Accuracy: "
            f"{val_accuracy * 100:.2f}%, "
            f"LR: "
            f"{optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_train_loss = train_loss

            best_train_accuracy = (
                train_accuracy
            )

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        scheduler.step()

        if patience_counter >= patience:

            print("Early stopping.")

            break

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accuracy": train_accuracies,
        "val_accuracy": val_accuracies
    }

    return (
        best_val_accuracy,
        best_train_loss,
        best_train_accuracy,
        best_model_state,
        history
    )
