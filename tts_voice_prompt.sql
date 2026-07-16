CREATE TABLE IF NOT EXISTS `tts_voice_prompt` (
    `spk_id` VARCHAR(64) NOT NULL,
    `name` VARCHAR(255) NULL,
    `voice_name` VARCHAR(255) NOT NULL,
    `model_name` VARCHAR(1024) NOT NULL DEFAULT '',
    `ref_audio_path` VARCHAR(1024) NULL,
    `ref_text` TEXT NULL,
    `prompt_blob` LONGBLOB NULL,
    `enabled` TINYINT(1) NOT NULL DEFAULT 1,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`spk_id`),
    KEY `idx_voice_name` (`voice_name`),
    KEY `idx_enabled` (`enabled`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
