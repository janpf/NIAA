num_epochs = 10
batch_size = 2
num_batches_per_pseudo_epoch = 2

batched_dataloader = []
i = 0
for j in range(0, 20, batch_size):
    batched_dataloader.append(list(range(j, j + batch_size)))

print("batched data:", batched_dataloader)

for epoch in range(num_epochs):
    current_epoch = iter(batched_dataloader)
    print("new (true) epoch:", epoch)
    for pseudo_epoch in range(len(batched_dataloader) // num_batches_per_pseudo_epoch):
        print("new pseudo epoch:", pseudo_epoch)
        for step in range(num_batches_per_pseudo_epoch):
            print("step in pseudo epoch:", step, "with batch:", next(current_epoch))
        print("validation here, at the end of each pseudo epoch")
