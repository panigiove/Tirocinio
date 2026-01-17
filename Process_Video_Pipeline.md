Piccolo file descrittivo di "process_video.py", "iou_tracker.py" e "appearence_utils.py"

# Pipeline di "process_video.py" in breve:

- inizializza il modello da usare con SAHI

- Inizializza il Tracker (IoU):
    Il tracker si basa sull'algoritmo Hungarian per determinare quali nuove detection si abbinano meglio alle detection vecchie, in modo così da tracciare una detection nel tempo

- crea la maschera da applicare al video (che in questo caso è trapezioidale)

- apre il video in input e lo analizza frame per frame:
    - applica la maschera al frame e utilizza YOLO con SAHI per fare la detection dei giocatori

    - per ogni prediction:
        - se il suo score supera "sahi_conf_threshold", allora ricava la bbox dove è presente la detection nell'immagine e chiama "compute_team_appearence()":
            nella sezione evidenziata dalla bbox estrae un crop, converte il crop in HSV, calcola l'istogramma con i parametri H e V, normalizza l'istogramma, converte il crop in Lab e estrae L_mean, concatena l'istrogramma e L_mean in un array e lo ritorna dopo averlo normalizzato con norma L2

        - procede poi a fare pose estimation sulla bbox:
            - se non rileva qualcosa con le impostazioni iniziali, prova a rifare pose estimation con parametri molto più permissivi, e procede poi normalmente
        
        - chiama "keypoints_to_pose_vec()", che rende disponibili i keypoints per fare matching pose-based con il tracker, e salva la posa e le altre informazioni della detection (appearence, bbox, keypoints, pose_vec)

    - aggiorna le informazioni dentro il tracker:
        se non ci sono tracciamenti, li inizializza, altrimenti se non ci sono proprio detection, tutti i tracciamenti aumentano il parametro 'missed' di uno, e se questo supera 50 (50 frame di tolleranza), vengono eliminati dal tracker. 
        Altrimenti viene inizializzata la matrice di costo usando come pesi le appearence, le pose e iou, e poi su di essa viene applicato l'algoritmo Hungarian. Se viene trovata una corrispondenza accettabile tra una detection nel tracker e una nuova, si aggiorna il tracker con quella nuova, altrimenti se non si è trovata corrispondenza viene creata una tracciatura per la detection. Infine tutte le detection tracciate e non aggiornate aumentano la propria variabile 'missed'. Come prima se essa supera 50, la traccia viene eliminata.
    - stampa sul frame tutte le informazioni delle varie pose e detection che il programma ha rilevato su quel frame. salva il frame in output in un video

- termina il programma

