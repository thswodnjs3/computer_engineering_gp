import os
import pandas as pd
import pickle
import torch
import torchvision.transforms as transforms

from itertools import combinations
from IPython.display import display
from tqdm.notebook import tqdm

from src.module import MyModel, Network
from src.utils import img_load, make_paths, make_dataset
from src.variable import classes, A_list, size, device, score_function

train_paths, test_paths = make_paths(path="./open(original)", mode="image")
mask_paths = make_paths(path="./open(original)", mode="mask")

# def find_size_batch():
#     test = MyModel()
#     result = []
#     start = 2
#     final_batch = 0
#     for idx, size in tqdm(enumerate(range(1024, 127, -32))):
#         final_size = size
#         for b in range(start,65,2):
#             test.setting(size=size, class_="bottle", batch_size=b)
            
#             if idx==0:
#                 model_ = test.make_model()
#             else:
#                 test.make_model(model_)
            
#             if test.train():
#                 final_batch = b
#                 start = b
#             else:
#                 break
#         print(f"Size: {final_size}, Batch: {final_batch}")
#         result.append((final_size, final_batch))
#     return result

# def modeling(class_, size, batch_size, model=None, augmentation=None, epochs=10, verbose=True, show_predict=False):
#     test = MyModel()
#     test.setting(size=size, class_=class_, augmentation=augmentation, batch_size=batch_size)
#     test.make_model(model)
#     test.train(epochs, verbose=verbose, show_predict=show_predict)
#     return test

def make_dataloader(mode, size, batch_size, class_=None, toTensor=True, normalize=[(0.3, 0.3, 0.3), (0.3, 0.3, 0.3)],
                    augmentation=None,augmented_report=False, augmented_target='bad', augmented_ratio=0, show_dataratio=False, test_aug=False, weighted_loss=False):
    train_loader, test_loader, num_classes, num_to_label, samples = \
        make_dataset(mode, size=size, train_paths=train_paths, test_paths=test_paths, class_=class_, batch_size=batch_size, toTensor=toTensor, normalize=normalize,
                     augmentation=augmentation, augmented_report=augmented_report, augmented_target=augmented_target, augmented_ratio=augmented_ratio,
                     show_dataratio=show_dataratio, test_aug=test_aug, weighted_loss=weighted_loss)
    return train_loader, test_loader, num_classes, num_to_label, samples
        
def make_model(class_, train_loader, test_loader, num_classes, num_to_label, mask_on=False):
    net = Network(num_classes=num_classes)
    model_ = MyModel(class_, train_loader, test_loader, num_to_label)
    model_.set_model(net, mask_on=mask_on)
    return model_

def train_model(model, epochs, lr=1e-4,
                criterion='CE', scaler='GS', optimizer='default',
                verbose=True, show_predict=False, show_cf_matrix=False, visualize=False, return_score=False, print_last=False,
                weighted_loss=False, samples=[], bad_strength=2, good_penalty=0.5,
                mask_on=False):
    score = model.train(epochs=epochs, verbose=verbose,
                criterion=criterion, scaler=scaler, optimizer=optimizer, lr=lr, show_predict=show_predict, show_cf_matrix=show_cf_matrix, visualize=visualize,
                return_score=return_score, print_last=print_last, weighted_loss=weighted_loss, samples=samples, bad_strength=bad_strength, good_penalty=good_penalty,
                mask_on=mask_on, mask_data=mask_paths)
    return model, score

def handler(augmentation=[], aug_combination=False, aug_each=True, existed_scores=None, exceptions=[], only=[],
            dataset_mode='good', img_size=size, batch_size=16, toTensor=True, normalize=[1, 1],
            augmented_report=True, augmented_target='bad', augmented_ratio=0, show_dataratio=False, test_aug=False,
            epochs=25, lr=1e-4, criterion='CE', scaler='GS', optimizer='default',
            verbose=False, show_predict=False, show_cf_matrix=False, visualize=False, return_score=True, print_last=True,
            final=False, show_difference=False, weighted_loss=False, bad_strength=2, good_penalty=0.5,
            mask_on=False):
    finals = []
    
    if len(augmentation)!=0:
        target = augmentation
    if aug_combination:
        target += ['+'.join(x) for i in range(2, len(augmentation)) for x in list(combinations(target, i))]

    if existed_scores is None:
        scores = {}
    
    m = normalize[0]
    s = normalize[1]
    
    if dataset_mode=='class':
        if len(augmentation)!=0:
            aug = [A_list[t] for t in target]
        else:
            aug=None
        train_loader, test_loader, num_classes, num_to_label, samples = make_dataloader(
            mode=dataset_mode, class_=None, size=img_size, batch_size=batch_size, toTensor=toTensor, normalize=[(m, m, m), (s, s, s)],
            augmentation=aug,
            augmented_report=augmented_report, augmented_target=augmented_target, augmented_ratio=augmented_ratio, show_dataratio=show_dataratio, test_aug=test_aug)
        model_ = make_model(None, train_loader, test_loader, num_classes, num_to_label, mask_on=False)
        model_, score = train_model(model_, epochs=epochs, lr=lr, criterion=criterion, scaler=scaler, optimizer=optimizer,
                            verbose=verbose, show_predict=show_predict, show_cf_matrix=show_cf_matrix, visualize=visualize, return_score=return_score, print_last=print_last)
        finals.append((dataset_mode, None, model_))
    else:
        for c in classes:
            if c not in only and len(only) != 0:
                continue
            
            if c in exceptions:
                continue
            
            if dataset_mode=='state' and c=='toothbrush' and final:
                continue
            
            if existed_scores is None:
                scores[c] = {}
            else:
                scores = existed_scores
            
            if aug_each:
                for t in target:
                    # print(f'Augmentation: {t}')
                    scores[c][t] = 0
                    
                    if t=='no_augmentation':
                        aug = None
                    elif '+' in t:
                        aug = [A_list[tt] for tt in t.split('+')]
                    else:
                        aug = [A_list[t]]
                        
                    train_loader, test_loader, num_classes, num_to_label, samples = make_dataloader(
                        mode=dataset_mode, class_=c, size=img_size, batch_size=batch_size, toTensor=toTensor, normalize=[(m, m, m), (s, s, s)],
                        augmentation=aug,
                        augmented_report=augmented_report, augmented_target=augmented_target, augmented_ratio=augmented_ratio, show_dataratio=show_dataratio, test_aug=test_aug,
                        weighted_loss=weighted_loss)
                    model_ = make_model(c, train_loader, test_loader, num_classes, num_to_label, mask_on=mask_on)
                    model_, score = train_model(model_, epochs=epochs, lr=lr, criterion=criterion, scaler=scaler, optimizer=optimizer,
                                        verbose=verbose, show_predict=show_predict, show_cf_matrix=show_cf_matrix, visualize=visualize, return_score=return_score, print_last=print_last,
                                        weighted_loss=weighted_loss, samples=samples, bad_strength=bad_strength, good_penalty=good_penalty,
                                        mask_on=mask_on)
                    scores[c][t] = score
            else:
                start = []
                start += [a for a in augmentation]
                t = ', '.join(start)
                # print(f"Augmentation: {t}")
                scores[c][t] = 0
                aug = [A_list[tt] for tt in augmentation if '+' not in tt]
                aug += [A_list[ttt] for tt in augmentation if '+' in tt for ttt in tt.split('+')]
                train_loader, test_loader, num_classes, num_to_label, samples = make_dataloader(
                    mode=dataset_mode, class_=c, size=img_size, batch_size=batch_size, toTensor=toTensor, normalize=[(m, m, m), (s, s, s)],
                    augmentation=aug,
                    augmented_report=augmented_report, augmented_target=augmented_target, augmented_ratio=augmented_ratio, show_dataratio=show_dataratio, test_aug=test_aug,
                    weighted_loss=weighted_loss)
                model_ = make_model(c, train_loader, test_loader, num_classes, num_to_label, mask_on=mask_on)
                model_, score = train_model(model_, epochs=epochs, lr=lr, criterion=criterion, scaler=scaler, optimizer=optimizer,
                                    verbose=verbose, show_predict=show_predict, show_cf_matrix=show_cf_matrix, visualize=visualize, return_score=return_score, print_last=print_last,
                                    weighted_loss=weighted_loss, samples=samples, bad_strength=bad_strength, good_penalty=good_penalty)
                scores[c][t] = score
            
            if show_difference:
                temp = pd.DataFrame(columns=list(scores[c].keys()), index=['score', 'change'])
                temp.columns.name = c
                temp.loc['score', :] = list(scores[c].values())
                temp.loc['change', :] = list(scores[c].values()) - list(scores[c].values())[0]
                display(temp.style.set_precision(3)\
                            .highlight_max(axis=1, color='red'))
            finals.append((dataset_mode, c, model_))
    if final==False:
        return model_, scores
    else:
        return finals
        
def remove_aug_scores(scores, class_):
    remove = [key for key in scores[class_].keys() if key!='no_augmentation']
    for k in remove:
        del(scores[class_][k])
    return scores

def final_test(finals, test_data, mode, ground_truth=None, file_name=None):
    if os.path.isfile(f'./pickle/{file_name}-temp.pickle') and file_name is not None:
        with open(f'./pickle/{file_name}-temp.pickle', 'rb') as fr:
            answer = pickle.load(fr)
    else:
        img_totensor = transforms.ToTensor()
        img_normalized = transforms.Normalize([1, 1, 1], [1, 1, 1])
        
        class_target = []
        good_target = []
        state_target = []
        
        class_model = finals['class']
        class_model.model.to(torch.device('cpu'))
        class_model.model.eval()
        for x in test_data:
            x = img_normalized(img_totensor(x))
            class_pred = class_model.model(torch.unsqueeze(x, 0))
                
            class_pred = class_pred.argmax(1).detach().cpu().numpy().tolist()
            
            class_target += [class_model.num_to_label[p] for p in class_pred]
            
            for p in class_pred:
                good_model = finals['good'][class_model.num_to_label[p]]
                good_model.model.to(torch.device('cpu'))
                good_model.model.eval()
                good_pred = good_model.model(torch.unsqueeze(x, 0))
                
                good_pred = good_pred.argmax(1).detach().cpu().numpy().tolist()
                good_target += [good_model.num_to_label[good_pred[0]]]
                
                if class_model.num_to_label[p]=='toothbrush':
                    state_target += ['None']
                    continue
                if good_model.num_to_label[good_pred[0]]=='bad':
                    state_model = finals['state'][class_model.num_to_label[p]]
                    state_model.model.to(torch.device('cpu'))
                    state_model.model.eval()
                    state_pred = state_model.model(torch.unsqueeze(x, 0))
                    
                    state_pred = state_pred.argmax(1).detach().cpu().numpy().tolist()
                    state_target += [state_model.num_to_label[state_pred[0]]]
                else:
                    state_target += ['good']
                    
        answer = []
        for c, g, s in zip(class_target, good_target, state_target):
            if g=='good':
                answer.append(f'{c}-{g}')
            else:
                if c=='toothbrush':
                    answer.append(f'{c}-defective')
                else:
                    answer.append(f'{c}-{s}')
        with open(f'./pickle/{file_name}-temp.pickle', 'wb') as fw:
            pickle.dump(answer, fw)
    if mode=='practice':
        class_ground = [g.split('-')[0] for g in ground_truth]
        class_answer = [a.split('-')[0] for a in answer]
        class_score = score_function(class_ground, class_answer)
        class_correct = [(g, a) for g, a in zip(ground_truth, answer) if g.split('-')[0]==a.split('-')[0]]
        
        good_ground = [g.split('-')[1] if g.split('-')[1]=='good' else 'bad' for g, a in class_correct]
        good_answer = [a.split('-')[1] if a.split('-')[1]=='good' else 'bad' for g, a in class_correct]
        good_score = score_function(good_ground, good_answer)
        
        good_correct = [(g, a) for g, a in zip(good_ground, good_answer) if g==a]
        bad_correct = [(g, a) for g, a in zip(good_ground, good_answer) if g=='bad' and a=='bad']
        
        state_ground = [g.split('-')[1] for g, a in class_correct if g.split('-')[1]!='good' and a.split('-')[1]!='good']
        state_answer = [a.split('-')[1] for g, a in class_correct if g.split('-')[1]!='good' and a.split('-')[1]!='good']
        state_score = score_function(state_ground, state_answer)
        state_correct = [(g, a) for g, a in zip(state_ground, state_answer) if g==a]
        
        print(f'Class: {class_score:.4f}({len(class_correct)}/{len(ground_truth)}), Good: {good_score:.4f}({len(good_correct)}/{len(class_correct)}), State: {state_score:.4f}({len(state_correct)}/{len(bad_correct)})')
        
        with open(f'./txt/{file_name}-class.txt', 'w') as f:
            f.write(f'{class_score}\n')
            for idx, (g, a) in enumerate(zip(class_ground, class_answer)):
                f.write(f'{idx}, {g}, {a}\n')
        
        with open(f'./txt/{file_name}-good.txt', 'w') as f:
            f.write(f'{good_score}\n')
            for idx, (g, a) in enumerate(zip(good_ground, good_answer)):
                f.write(f'{idx}, {g}, {a}\n')
        
        with open(f'./txt/{file_name}-state.txt', 'w') as f:
            f.write(f'{state_score}\n')
            for idx, (g, a) in enumerate(zip(state_ground, state_answer)):
                f.write(f'{idx}, {g}, {a}\n')
    return answer

def make_ground_truth(size):
    data = sorted([(t.split('/')[5], t.split('/')[2], t.split('/')[4], t) for t in test_paths], key=lambda x: x[0])
    data += sorted([(t.split('/')[5], t.split('/')[2], t.split('/')[4], t) for t in test_paths], key=lambda x: x[0])
    data = [(img_load(p, (size, size)), f'{c}-{s}') if s!='good' else (img_load(p, (size, size)), f'{c}-good') for _, c, s, p in data]
    return data

def make_submission(size):
    t_df = pd.read_csv('./open/test_df.csv', index_col=0)
    data = [img_load(f'./open/test/{x}', (size, size)) for x in list(t_df['file_name'])]
    return data