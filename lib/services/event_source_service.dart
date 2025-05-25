export 'event_source_interface.dart';
export 'event_source_service_mobile.dart' // Default (mobile)
    if (dart.library.html) 'event_source_service_web.dart'; // Web