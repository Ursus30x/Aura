using UnityEngine;
using UMA;
using UMA.CharacterSystem;
using UMA.PoseTools;
using System;
using System.Globalization;

[Serializable]
public class RotationData {
    public float pitch;
    public float yaw;
    public float roll;
}

[Serializable]
public class AnimationPacket {
    public string type;
    public long timestamp;
    public RotationData rotation;
}

public class DirectBoneController : MonoBehaviour
{
    [Header("Network")]
    public NetworkManagerUDP udpReceiver;

    [Header("Configuration")]
    public float lerpSpeed = 25f;
    public bool applyRotation = true;

    [Header("Head Rotation")]
    [Tooltip("Per-axis gain (pitch, yaw, roll). Flip sign if an axis moves the wrong way.")]
    public Vector3 rotationSensitivity = new Vector3(1f, 1f, 1f);
    [Tooltip("Static offset added after sensitivity (degrees).")]
    public Vector3 rotationOffset = Vector3.zero;
    [Tooltip("Max deviation from rest per axis (degrees). Stops noise spikes going insane.")]
    public Vector3 maxAngle = new Vector3(40f, 55f, 35f);
    [Tooltip("Head rotation smoothing speed.")]
    public float rotationLerpSpeed = 12f;

    [Header("Mouth")]
    [Tooltip("Gain on jaw open so the mouth opens as wide as the source intends.")]
    public float jawOpenGain = 1.7f;
    [Tooltip("Raw mediapipe jawOpen when the mouth is CLOSED. Mediapipe reports a non-zero baseline at rest, so subtract it to keep the resting mouth shut (deadzone).")]
    [Range(0f, 0.4f)] public float jawOpenBaseline = 0.1f;
    [Tooltip("Raw mediapipe jawOpen value treated as 'fully open' (AFTER subtracting the baseline). A relaxed-but-clearly-open mouth only reaches ~0.35-0.45, so this remaps it to a full open.")]
    public float jawOpenInputMax = 0.85f;
    [Tooltip("UMA jawOpen_Close value at full open. >1 forces a wide dentist-style gape (clamped to 1 by UMA).")]
    public float jawOpenRange = 2.0f;
    [Tooltip("Direction the jaw bone opens. +1 normally. If the avatar CLENCHES instead of opening (lockjaw), flip to -1 — the rig's jawOpen_Close is inverted.")]
    [Range(-1f, 1f)] public float jawOpenSign = 1f;
    [Tooltip("How strongly an open jaw cancels the smile, so a wide gape doesn't read as a grin.")]
    [Range(0f, 1f)] public float smileSuppressByJaw = 0.85f;
    [Tooltip("Extra lip/teeth spread on a wide gape (lower lip + upper lip). 0 = OFF (use this if the lips clench the teeth shut). UMA jaw pose clamps at 1, so this exposes teeth.")]
    [Range(0f, 2f)] public float jawGapeBoost = 0f;
    [Tooltip("Sign of the lip-gape push. +1 if a wide gape correctly parts the teeth, -1 if the lips move the wrong way and clamp the teeth shut on this rig.")]
    [Range(-1f, 1f)] public float jawGapeSign = 1f;
    [Tooltip("Gain on lip pucker (dzióbek). Mediapipe pucker values are small, so amplify.")]
    public float puckerGain = 2.5f;
    [Tooltip("Gain on tongue out. Mediapipe tongueOut peaks low, so amplify to make the tongue actually protrude.")]
    public float tongueOutGain = 1.6f;

    [Header("Expressiveness")]
    [Tooltip("Global multiplier on all facial expressions (eyes, mouth, brows, cheeks, nose). >1 = livelier.")]
    public float expressionGain = 1.5f;
    [Tooltip("Response curve. >1 boosts small/subtle motions so faint expressions read clearly; 1 = linear.")]
    [Range(1f, 3f)] public float expressionCurve = 1.6f;
    [Tooltip("Snappiness of expression smoothing. Higher = sharper, more alive; lower = softer.")]
    public float expressionLerpSpeed = 0f;
    [Tooltip("Separate gain for eye open/close. Lower than expressionGain so the avatar doesn't squint/over-blink.")]
    [Range(0f, 2f)] public float eyeGain = 0.7f;

    [Header("Neck")]
    [Tooltip("Drive the neck bone with a fraction of the head rotation for a natural look (0 = head only).")]
    [Range(0f, 1f)] public float neckFollow = 0.4f;

    [Header("Calibration")]
    [Tooltip("Apply the one-shot calibration packet (face DNA + skin color) sent by the CV pipeline.")]
    public bool applyCalibration = true;
    [Tooltip("Multipliers on the raw calibration ratios before they are written to UMA DNA.")]
    public float noseWidthScale = 1.5f;
    public float mouthSizeScale = 1.2f;
    public float eyeSpacingScale = 2.0f;

    private UMAData umaData;
    private UMAExpressionPlayer expressionPlayer;
    private DynamicCharacterAvatar avatar;
    private RotationData currentRotation = new RotationData();
    private int headHash = 0;
    private bool initialized = false;
    private Quaternion restHeadRot = Quaternion.identity;
    private Quaternion smoothedHeadRot = Quaternion.identity;
    private int neckHash = 0;
    private Quaternion restNeckRot = Quaternion.identity;
    private Quaternion smoothedNeckRot = Quaternion.identity;

    void Start()
    {
        avatar = GetComponent<DynamicCharacterAvatar>();
        if (udpReceiver == null) udpReceiver = UnityEngine.Object.FindAnyObjectByType<NetworkManagerUDP>();
        
        if (avatar != null) {
            avatar.CharacterUpdated.AddListener(OnUmaCharacterUpdated);
        }
    }

    private void OnUmaCharacterUpdated(UMAData obj)
    {
        umaData = obj;
        expressionPlayer = GetComponent<UMAExpressionPlayer>();
        if (expressionPlayer == null) expressionPlayer = GetComponentInChildren<UMAExpressionPlayer>();

        if (umaData != null && umaData.animator != null) {
            Transform head = umaData.animator.GetBoneTransform(HumanBodyBones.Head);
            if (head != null) {
                headHash = UMAUtils.StringToHash(head.name);
                // Capture the rig's neutral head pose so tracked rotation is applied as a delta
                // on top of it instead of overwriting it (which caused the head to snap/twist).
                restHeadRot = umaData.skeleton.GetRotation(headHash);
                smoothedHeadRot = restHeadRot;
            }

            Transform neck = umaData.animator.GetBoneTransform(HumanBodyBones.Neck);
            if (neck != null) {
                neckHash = UMAUtils.StringToHash(neck.name);
                restNeckRot = umaData.skeleton.GetRotation(neckHash);
                smoothedNeckRot = restNeckRot;
            }
        }

        if (expressionPlayer != null) {
            expressionPlayer.enableBlinking = false;
            expressionPlayer.enableSaccades = false;
            expressionPlayer.processing = true;
        }

        initialized = true;
    }

    void Update()
    {
        if (!initialized || udpReceiver == null) return;

        string json = udpReceiver.GetNextPacket();
        while (json != null) {
            ProcessPacket(json);
            json = udpReceiver.GetNextPacket();
        }
    }

    private void ForceAnimationOverride()
    {
        expressionPlayer.overrideMecanimJaw = true;
        expressionPlayer.overrideMecanimNeck = true;
        expressionPlayer.overrideMecanimHead = true;
        expressionPlayer.overrideMecanimEyes = true;
        expressionPlayer.overrideMecanimHands = true;
    }

    private void ProcessPacket(string json)
    {
        ForceAnimationOverride();

        if (json.Contains("\"type\": \"animation\"")) {
            var packet = JsonUtility.FromJson<AnimationPacket>(json);
            currentRotation = packet.rotation;

            if (expressionPlayer != null) {
                expressionPlayer.processing = true;

                // NORMALIZACJA: 
                // Funkcja lokalna dla par (np. Smile - Frown). Wynik od -1 do 1.
                float Pair(string pos, string neg) => GetJsonFloat(json, pos) - GetJsonFloat(json, neg);

                // EKSPRESYWNOŚĆ: rozjaśnij małe ruchy (krzywa) i wzmocnij (gain), zachowując znak. Zatrzaskuj do [-1,1].
                // Mediapipe rzadko dochodzi do 1.0, więc bez tego mimika wygląda blado względem modela.
                float Ex(float v) {
                    float a = Mathf.Pow(Mathf.Clamp01(Mathf.Abs(v)), 1f / expressionCurve) * expressionGain;
                    return Mathf.Clamp(Mathf.Sign(v) * a, -1f, 1f);
                }

                // --- MAPOWANIE WSZYSTKICH EKSPRESJI UMA ---
                // Use a dedicated, snappier lerp for expressions so peaks aren't smoothed away (falls back to lerpSpeed if 0).
                float dtLerp = Time.deltaTime * (expressionLerpSpeed > 0f ? expressionLerpSpeed : lerpSpeed);

                // Unidirectional blendshape (0..1, neutral = 0). Map direct, NOT (val*2)-1,
                // otherwise the rest pose sits at -1 (clenched) and the mouth barely opens.
                float Uni(string key, float gain = 1f) => Mathf.Clamp(GetJsonFloat(json, key) * gain, 0f, 1f);

                // SZCZĘKA (Jaw) — przywrócona działająca formuła (BEZ baseline — to psuło ruch).
                // Normalizuj surowy jawOpen do 0..1, potem napędzaj jawOpen_Close. jawOpenSign odwraca kierunek
                // jeśli rig zaciska zamiast otwierać (SZCZĘKOŚCISK = znak odwrotny → ustaw jawOpenSign na -1).
                float jawNorm = Mathf.Clamp01(GetJsonFloat(json, "jawOpen") * jawOpenGain / Mathf.Max(0.01f, jawOpenInputMax));
                expressionPlayer.jawOpen_Close = Mathf.Lerp(expressionPlayer.jawOpen_Close, Mathf.Clamp(jawNorm * jawOpenRange * jawOpenSign, -1f, 1f), dtLerp);
                expressionPlayer.jawForward_Back = Mathf.Lerp(expressionPlayer.jawForward_Back, Uni("jawForward"), dtLerp);
                expressionPlayer.jawLeft_Right = Mathf.Lerp(expressionPlayer.jawLeft_Right, Pair("jawRight", "jawLeft"), dtLerp);

                // OCZY (Eyes)
                // Eyes use a gentler gain than the rest so the avatar doesn't squint/over-blink.
                expressionPlayer.leftEyeOpen_Close = Mathf.Lerp(expressionPlayer.leftEyeOpen_Close, Mathf.Clamp(Pair("eyeWideLeft", "eyeBlinkLeft") * eyeGain, -1f, 1f), dtLerp);
                expressionPlayer.rightEyeOpen_Close = Mathf.Lerp(expressionPlayer.rightEyeOpen_Close, Mathf.Clamp(Pair("eyeWideRight", "eyeBlinkRight") * eyeGain, -1f, 1f), dtLerp);
                expressionPlayer.leftEyeUp_Down = Mathf.Lerp(expressionPlayer.leftEyeUp_Down, Pair("eyeLookUpLeft", "eyeLookDownLeft"), dtLerp);
                expressionPlayer.rightEyeUp_Down = Mathf.Lerp(expressionPlayer.rightEyeUp_Down, Pair("eyeLookUpRight", "eyeLookDownRight"), dtLerp);
                expressionPlayer.leftEyeIn_Out = Mathf.Lerp(expressionPlayer.leftEyeIn_Out, Pair("eyeLookInLeft", "eyeLookOutLeft"), dtLerp);
                expressionPlayer.rightEyeIn_Out = Mathf.Lerp(expressionPlayer.rightEyeIn_Out, Pair("eyeLookInRight", "eyeLookOutRight"), dtLerp);

                // USTA (Mouth)
                expressionPlayer.mouthLeft_Right = Mathf.Lerp(expressionPlayer.mouthLeft_Right, Ex(Pair("mouthRight", "mouthLeft")), dtLerp);
                expressionPlayer.mouthUp_Down = Mathf.Lerp(expressionPlayer.mouthUp_Down, Ex(Pair("mouthShrugUpper", "mouthLowerDownLeft")), dtLerp);
                expressionPlayer.mouthNarrow_Pucker = Mathf.Lerp(expressionPlayer.mouthNarrow_Pucker, Mathf.Clamp(GetJsonFloat(json, "mouthFunnel") - GetJsonFloat(json, "mouthPucker") * puckerGain, -1f, 1f), dtLerp);
                // A wide gape pulls lip corners and registers as a smile in mediapipe; fade smile out while jaw is open.
                float smileScale = 1f - jawNorm * smileSuppressByJaw;
                expressionPlayer.leftMouthSmile_Frown = Mathf.Lerp(expressionPlayer.leftMouthSmile_Frown, Ex(Pair("mouthSmileLeft", "mouthFrownLeft")) * smileScale, dtLerp);
                expressionPlayer.rightMouthSmile_Frown = Mathf.Lerp(expressionPlayer.rightMouthSmile_Frown, Ex(Pair("mouthSmileRight", "mouthFrownRight")) * smileScale, dtLerp);
                
                // WARGI (Lips)
                // Zęby rozsuwa jawOpen_Close (kość szczęki) powyżej. Tutaj tylko POMAGAMY ustom rozśliwić się
                // za kością szczęki: dolna warga w dół, górna w górę, LINIOWO z jawNorm (bez krzywej = bez uncanny).
                // jawGapeSign odwraca kierunek jeśli rig pcha wargi nie w tę stronę.
                float gape = jawNorm * jawGapeBoost * jawGapeSign;
                expressionPlayer.leftLowerLipUp_Down = Mathf.Lerp(expressionPlayer.leftLowerLipUp_Down, Mathf.Clamp(Pair("mouthShrugLower", "mouthLowerDownLeft") - gape, -1f, 1f), dtLerp);
                expressionPlayer.rightLowerLipUp_Down = Mathf.Lerp(expressionPlayer.rightLowerLipUp_Down, Mathf.Clamp(Pair("mouthShrugLower", "mouthLowerDownRight") - gape, -1f, 1f), dtLerp);
                expressionPlayer.leftUpperLipUp_Down = Mathf.Lerp(expressionPlayer.leftUpperLipUp_Down, Mathf.Clamp(Uni("mouthUpperUpLeft") + gape, -1f, 1f), dtLerp);
                expressionPlayer.rightUpperLipUp_Down = Mathf.Lerp(expressionPlayer.rightUpperLipUp_Down, Mathf.Clamp(Uni("mouthUpperUpRight") + gape, -1f, 1f), dtLerp);

                // BRWI (Brows)
                expressionPlayer.browsIn = Mathf.Lerp(expressionPlayer.browsIn, Ex((GetJsonFloat(json, "browDownLeft") + GetJsonFloat(json, "browDownRight")) / 2f), dtLerp);
                expressionPlayer.midBrowUp_Down = Mathf.Lerp(expressionPlayer.midBrowUp_Down, Ex(Uni("browInnerUp")), dtLerp);
                expressionPlayer.leftBrowUp_Down = Mathf.Lerp(expressionPlayer.leftBrowUp_Down, Ex(Pair("browOuterUpLeft", "browDownLeft")), dtLerp);
                expressionPlayer.rightBrowUp_Down = Mathf.Lerp(expressionPlayer.rightBrowUp_Down, Ex(Pair("browOuterUpRight", "browDownRight")), dtLerp);

                // POLICZKI I NOS (Cheeks & Nose)
                float globalPuff = GetJsonFloat(json, "cheekPuff");
                expressionPlayer.leftCheekPuff_Squint = Mathf.Lerp(expressionPlayer.leftCheekPuff_Squint, Ex(Pair("cheekPuff", "cheekSquintLeft")), dtLerp);
                expressionPlayer.rightCheekPuff_Squint = Mathf.Lerp(expressionPlayer.rightCheekPuff_Squint, Ex(Pair("cheekPuff", "cheekSquintRight")), dtLerp);
                expressionPlayer.noseSneer = Mathf.Lerp(expressionPlayer.noseSneer, Ex(Mathf.Max(GetJsonFloat(json, "noseSneerLeft"), GetJsonFloat(json, "noseSneerRight"))), dtLerp);

                // JĘZYK (Tongue)
                // Mediapipe only provides tongueOut; the other tongue axes have no source data, keep them at 0.
                expressionPlayer.tongueOut = Mathf.Lerp(expressionPlayer.tongueOut, Mathf.Clamp01(GetJsonFloat(json, "tongueOut") * tongueOutGain), dtLerp);
                expressionPlayer.tongueCurl = Mathf.Lerp(expressionPlayer.tongueCurl, 0f, dtLerp);
                expressionPlayer.tongueUp_Down = Mathf.Lerp(expressionPlayer.tongueUp_Down, 0f, dtLerp);
                expressionPlayer.tongueLeft_Right = Mathf.Lerp(expressionPlayer.tongueLeft_Right, 0f, dtLerp);
                expressionPlayer.tongueWide_Narrow = Mathf.Lerp(expressionPlayer.tongueWide_Narrow, 0f, dtLerp);
            }
        }
        else if (json.Contains("\"type\": \"calibration\"")) {
            if (applyCalibration) ApplyCalibration(json);
        }
    }

    private void ApplyCalibration(string json)
    {
        if (avatar == null) return;
        var dna = avatar.GetDNA();
        if (dna == null) return;

        // Map the CV pipeline's facial topography ratios onto UMA DNA sliders.
        if (dna.ContainsKey("noseWidth") && json.Contains("nose_width"))
            dna["noseWidth"].Set(Mathf.Clamp01(GetJsonFloat(json, "nose_width") * noseWidthScale));
        if (dna.ContainsKey("mouthSize") && json.Contains("mouth_width"))
            dna["mouthSize"].Set(Mathf.Clamp01(GetJsonFloat(json, "mouth_width") * mouthSizeScale));
        if (dna.ContainsKey("eyeSpacing") && json.Contains("inter_eye_dist"))
            dna["eyeSpacing"].Set(Mathf.Clamp01(GetJsonFloat(json, "inter_eye_dist") * eyeSpacingScale));

        // Skin color: "skin_color":[r,g,b] (0-255).
        const string skinPattern = "\"skin_color\":[";
        int start = json.IndexOf(skinPattern);
        if (start != -1) {
            start += skinPattern.Length;
            int end = json.IndexOf(']', start);
            if (end > start) {
                string[] rgb = json.Substring(start, end - start).Split(',');
                if (rgb.Length == 3
                    && int.TryParse(rgb[0].Trim(), out int r)
                    && int.TryParse(rgb[1].Trim(), out int g)
                    && int.TryParse(rgb[2].Trim(), out int b)) {
                    avatar.SetColor("Skin", new Color(r / 255f, g / 255f, b / 255f));
                }
            }
        }

        // Rebuild once to bake the DNA + color changes (calibration is a one-shot packet, not per-frame).
        avatar.BuildCharacter();
    }

    void LateUpdate()
    {
        if (!initialized || umaData == null || !applyRotation) return;

        if (headHash != 0) {
            // Apply tracked head pose as a clamped delta on top of the rig's rest pose,
            // then smooth toward it so noise spikes don't snap the head around.
            // Source pitch/yaw are swapped relative to UMA's bone axes, so map yaw->X(pitch) and pitch->Y(yaw).
            float pitch = Mathf.Clamp(currentRotation.yaw   * rotationSensitivity.x + rotationOffset.x, -maxAngle.x, maxAngle.x);
            float yaw   = Mathf.Clamp(currentRotation.pitch * rotationSensitivity.y + rotationOffset.y, -maxAngle.y, maxAngle.y);
            float roll  = Mathf.Clamp(-currentRotation.roll * rotationSensitivity.z + rotationOffset.z, -maxAngle.z, maxAngle.z);

            // Split the rotation: the neck carries a fraction, the head carries the remainder,
            // so the whole upper body reads as a natural turn instead of just the head pivoting.
            if (neckHash != 0 && neckFollow > 0f) {
                Quaternion neckTarget = restNeckRot * Quaternion.Euler(pitch * neckFollow, yaw * neckFollow, roll * neckFollow);
                smoothedNeckRot = Quaternion.Slerp(smoothedNeckRot, neckTarget, Time.deltaTime * rotationLerpSpeed);
                umaData.skeleton.SetRotation(neckHash, smoothedNeckRot);
            }

            float headShare = (neckHash != 0) ? (1f - neckFollow) : 1f;
            Quaternion targetRot = restHeadRot * Quaternion.Euler(pitch * headShare, yaw * headShare, roll * headShare);
            smoothedHeadRot = Quaternion.Slerp(smoothedHeadRot, targetRot, Time.deltaTime * rotationLerpSpeed);
            umaData.skeleton.SetRotation(headHash, smoothedHeadRot);
        }
    }

    private float GetJsonFloat(string json, string key) {
        string pattern = "\"" + key + "\":";
        int start = json.IndexOf(pattern);
        if (start == -1) return 0f;
        start += pattern.Length;
        while (start < json.Length && char.IsWhiteSpace(json[start])) start++;
        int end = start;
        while (end < json.Length && (char.IsDigit(json[end]) || json[end] == '.' || json[end] == '-' || json[end] == 'e' || json[end] == 'E' || json[end] == '+')) {
            end++;
        }
        if (start == end) return 0f;
        string valStr = json.Substring(start, end - start);
        if (float.TryParse(valStr, NumberStyles.Float, CultureInfo.InvariantCulture, out float result)) return result;
        return 0f;
    }
}
